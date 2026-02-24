#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/11/06
# @Desc: RePRMeta — Retrieval + Prompted Generation with Reptile-style Meta
#       - inputs: (video, audio, text[B,82,512])
#       - generator learns an offset Δ added to retrieved donors
#       - meta updates: within-batch (beta) + across-batch (gamma)
#       - evaluate uses complete_for_eval (一次补全)

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer

# ----------------------------- #
# Utils
# ----------------------------- #


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    """[B,T,C] -> [B,C] by mean; [B,C] stays."""
    return x.mean(dim=1) if x.dim() == 3 else x


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """a:[N,D], b:[M,D] -> [N,M]"""
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return a @ b.t()


def _interpolate_T(src: torch.Tensor, T: int) -> torch.Tensor:
    """Resize temporal dim to T. Supports [...,T,C]."""
    if src.size(-2) == T:
        return src.contiguous()
    x = src.transpose(-2, -1)  # [..., C, T]
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return x.transpose(-2, -1).contiguous()


def _ensure_bool_mask(
    masks: Optional[torch.Tensor], B: int, device: torch.device
) -> torch.Tensor:
    """Ensure masks is [B,3] bool on device; default all True."""
    if masks is None:
        return torch.ones((B, 3), dtype=torch.bool, device=device)
    if masks.device != device:
        masks = masks.to(device)
    if masks.dtype != torch.bool:
        masks = masks.bool()
    assert masks.shape == (B, 3), f"masks must be [B,3], got {tuple(masks.shape)}"
    return masks


def _sanitize_triplet(triplet):
    v, a, t = triplet

    def _clean(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()
        x = torch.clamp(x, -1e4, 1e4)
        return x.contiguous()

    return _clean(v), _clean(a), _clean(t)


def _resize_time(feat: torch.Tensor, tgt_T: int) -> torch.Tensor:
    """Resize temporal dimension to tgt_T using linear interpolation."""
    if feat.dim() == 3 and feat.size(0) == 1:
        feat = feat.squeeze(0)
    if feat.dim() == 1:
        feat = feat.unsqueeze(1)
    T, C = feat.shape
    if T == tgt_T:
        return feat
    x = feat.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x = F.interpolate(x, size=tgt_T, mode="linear", align_corners=False)
    return x.squeeze(0).transpose(0, 1).contiguous()


def _align_temporal(donor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align donor's temporal length and channel count to target."""
    if donor.dim() == 3 and donor.size(0) == 1:
        donor = donor.squeeze(0)
    if target.dim() == 3 and target.size(0) == 1:
        target = target.squeeze(0)
    Td, Cd = donor.shape[-2], donor.shape[-1]
    Tt, Ct = target.shape[-2], target.shape[-1]
    donor = _resize_time(donor, Tt)
    if Cd == Ct:
        return donor
    if Cd > Ct:
        return donor[..., :Ct]
    pad = donor.new_zeros((Tt, Ct - Cd))
    return torch.cat([donor, pad], dim=-1)


# ----------------------------- #
# Prompted Generator (predicts OFFSETS)
# ----------------------------- #


class _PromptMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Generator(nn.Module):
    """
    Prompted generator that predicts offsets Δv/Δa/Δt:
      sequence = [p_missing, p_task, v_tok, a_tok, t_tok, v_ret_tok, a_ret_tok, t_ret_tok]
      -> Transformer -> heads (offsets)
    """

    def __init__(
        self,
        v_dim: int,
        a_dim: int,
        t_dim: int,
        hid: int = 256,
        nhead: int = 8,
        nlayers: int = 2,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hid)
        self.a_proj = nn.Linear(a_dim, hid)
        self.t_proj = nn.Linear(t_dim, hid)
        self.vr_proj = nn.Linear(v_dim, hid)
        self.ar_proj = nn.Linear(a_dim, hid)
        self.tr_proj = nn.Linear(t_dim, hid)

        self.p_m = _PromptMLP(3, hid)
        self.p_t = _PromptMLP(n_tasks, hid)

        enc = nn.TransformerEncoderLayer(
            d_model=hid, nhead=nhead, dim_feedforward=hid * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)

        # heads output OFFSETS
        self.v_head = nn.Linear(hid, v_dim)
        self.a_head = nn.Linear(hid, a_dim)
        self.t_head = nn.Linear(hid, t_dim)

    def forward(
        self,
        v,
        a,
        t,
        miss_ind,
        task_1h,
        v_ret,
        a_ret,
        t_ret,
    ):
        vt, at, tt = (
            self.v_proj(_flatten_time(v)),
            self.a_proj(_flatten_time(a)),
            self.t_proj(_flatten_time(t)),
        )
        vrt, art, trt = (
            self.vr_proj(_flatten_time(v_ret)),
            self.ar_proj(_flatten_time(a_ret)),
            self.tr_proj(_flatten_time(t_ret)),
        )
        seq = torch.stack(
            [self.p_m(miss_ind), self.p_t(task_1h), vt, at, tt, vrt, art, trt], dim=1
        )  # [B,8,hid]
        h = self.encoder(seq)

        # 使用 donor token 的位置（5: v_ret, 6: a_ret, 7: t_ret）附近的聚合信息也能被注意力利用
        dv = self.v_head(h[:, 2, :])  # Δv 基于自身 token 的槽位
        da = self.a_head(h[:, 3, :])  # Δa
        dt = self.t_head(h[:, 4, :])  # Δt

        # 展开回时间维（恒定 offset across time）
        if v.dim() == 3:
            dv = dv.unsqueeze(1).expand(-1, v.size(-2), -1).contiguous()
        if a.dim() == 3:
            da = da.unsqueeze(1).expand(-1, a.size(-2), -1).contiguous()
        if t.dim() == 3:
            dt = dt.unsqueeze(1).expand(-1, t.size(-2), -1).contiguous()
        return dv, da, dt  # offsets


# ----------------------------- #
# RePRMeta (with Reptile-style Meta)
# ----------------------------- #


class RePROffset(ContinualModel):
    """
    Retrieval + Prompted Generation (offset) + Reptile-style Meta.

    - Input:
        inputs: (video, audio, text[B,82,512])
        masks:  [B,3] 1=present, 0=missing (missing parts are zeroed)
    - Strategy:
        If text is missing -> retrieval only
        If video/audio is missing -> (retrieval donor) + Δ (generator)
    - Evaluation:
        v_c, a_c, t_c = self.complete_for_eval((v, a, t), masks, task_id)
        outputs, _ = self.net((v_c, a_c, t_c), returnt="all")
    """

    NAME = "repr_offset"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        # Buffer
        self.buffer = Buffer(self.args.buffer_size, device="cpu")
        self.buffer.empty()

        # Dataset-based default dims
        self.v_dim = getattr(args, "video_dim", None)
        self.a_dim = getattr(args, "audio_dim", None)
        self.t_dim = getattr(args, "text_dim", None)  # 512

        ds = (getattr(args, "dataset", "") or "").lower()
        if self.v_dim is None or self.a_dim is None or self.t_dim is None:
            if "fs1000" in ds:
                self.v_dim = 768 if self.v_dim is None else self.v_dim
                self.a_dim = 768 if self.a_dim is None else self.a_dim
                self.t_dim = 512 if self.t_dim is None else self.t_dim
            elif ("rg" in ds) or ("fs" in ds):
                self.v_dim = 1024 if self.v_dim is None else self.v_dim
                self.a_dim = 768 if self.a_dim is None else self.a_dim
                self.t_dim = 512 if self.t_dim is None else self.t_dim

        # Retrieval / replay params
        self.retrieval_topk = getattr(self.args, "retrieval_topk", 3)
        self.sim_modal_weights = getattr(
            self.args, "sim_modal_weights", (5.0, 3.0, 1.0)
        )
        self.ret_lambda = getattr(
            self.args, "ret_lambda", 0.75
        )  # 已弃用（不再线性融合）

        # Meta params
        self.batch_num = getattr(self.args, "batch_num", 3)
        self.beta = getattr(self.args, "beta", 0.25)  # within-batch meta
        self.gamma = getattr(self.args, "gamma", 0.5)  # across-batch meta

        # Generator params
        self.hidden = getattr(args, "gen_hidden", 256)
        self.gen_rec_weight = getattr(args, "gen_rec_weight", 1.0)

        # Tasks (for prompt)
        self.n_tasks = self.args.n_tasks
        self.cur_task = 0

        # Modules
        self.generator = _Generator(
            self.v_dim, self.a_dim, self.t_dim, hid=self.hidden, n_tasks=self.n_tasks
        ).to(self.device)

        # Two optimizers: generator vs backbone
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.opt_backbone = torch.optim.Adam(
            self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # (optional) log
        if hasattr(self.args, "logging") and self.args.logging is not None:
            total_params = sum(p.numel() for p in self.generator.parameters())
            self.args.logging.info(
                f"[RePROffset] Generator (offset) params: {total_params}"
            )

    # ---------- helpers ----------
    def _indicators(self, masks: torch.Tensor, task_id: Optional[int] = None):
        B = masks.size(0)
        miss_ind = masks.float().to(self.device)
        if task_id is None:
            tid = min(self.cur_task, self.n_tasks - 1)
        else:
            tid = int(task_id)
            tid = 0 if tid < 0 else min(tid, self.n_tasks - 1)
        task = torch.zeros(B, self.n_tasks, device=self.device)
        task[:, tid] = 1.0
        return miss_ind, task

    def _snapshot(self):
        if hasattr(self, "module") and hasattr(self.module, "get_params"):
            with torch.no_grad():
                return self.module.get_params(device=self.device).clone().detach()
        return None

    def _meta_pull(self, base, coeff: float):
        if base is None or coeff <= 0 or not hasattr(self.module, "get_params"):
            return
        with torch.no_grad():
            curr = self.module.get_params(device=self.device)
            self.module.set_params(base + coeff * (curr - base))

    # ---------- ONE-SHOT completion for evaluation ----------
    def complete_for_eval(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: torch.Tensor,
        task_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(inputs, tuple) and len(inputs) == 3
        v, a, t = inputs
        v, a, t = v.to(self.device), a.to(self.device), t.to(self.device)
        B = v.size(0)
        masks = _ensure_bool_mask(masks, B, self.device)

        # 检索 donor
        v_ret, a_ret, t_ret = self._impute_missing_with_buffer((v, a, t), masks)

        with torch.no_grad():
            miss_ind, task_1h = self._indicators(masks, task_id=task_id)
            # 生成器预测 offset
            dv, da, dt = self.generator(v, a, t, miss_ind, task_1h, v_ret, a_ret, t_ret)

            # 时间维对齐
            if v.dim() == 3 and dv.size(-2) != v.size(-2):
                dv = _interpolate_T(dv, v.size(-2))
            if a.dim() == 3 and da.size(-2) != a.size(-2):
                da = _interpolate_T(da, a.size(-2))
            if t.dim() == 3 and dt.size(-2) != t.size(-2):
                dt = _interpolate_T(dt, t.size(-2))

            # 缺失处：donor + Δ；text 缺失仍仅用 donor（保守）
            bmv = (
                masks[:, 0].view(-1, 1, 1) if v.dim() == 3 else masks[:, 0].view(-1, 1)
            )
            bma = (
                masks[:, 1].view(-1, 1, 1) if a.dim() == 3 else masks[:, 1].view(-1, 1)
            )
            bmt = masks[:, 2].view(-1, 1, 1)  # text [B,82,512]

            v_fused = torch.where(bmv, v, v_ret + dv)
            a_fused = torch.where(bma, a, a_ret + da)
            t_fused = torch.where(bmt, t, t_ret)  # 不加 dt，稳定起见

            v_fused, a_fused, t_fused = _sanitize_triplet((v_fused, a_fused, t_fused))
            v_fused, a_fused, t_fused = (
                v_fused.detach(),
                a_fused.detach(),
                t_fused.detach(),
            )

        return v_fused.contiguous(), a_fused.contiguous(), t_fused.contiguous()

    # ---------- ONLY-BUFFER IMPUTATION (eval & training) ----------
    @torch.no_grad()
    def _impute_missing_with_buffer(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 buffer【全部样本】检索，对缺失模态进行补全；
        - 可用模态参与相似度（加权）
        - 缺失模态用 top-k donor 均值，并对齐时间/通道
        - text 始终保持 [B, 82, 512]
        """
        v, a, t = inputs
        device = self.device
        B = v.size(0)
        masks = masks.to(device)

        if self.buffer.is_empty():
            return inputs

        buf_all = self.buffer.get_all_data()
        buf_triplet = buf_all[0]  # ((v,a,t), labels, [task_labels])
        if not (isinstance(buf_triplet, tuple) and len(buf_triplet) == 3):
            return inputs

        buf_v, buf_a, buf_t = (x.to(device) for x in buf_triplet)
        N = buf_v.size(0)
        if N == 0:
            return inputs

        wv, wa, wt = self.sim_modal_weights
        buf_v_flat = buf_v.reshape(N, -1) * wv
        buf_a_flat = buf_a.reshape(N, -1) * wa
        buf_t_flat = buf_t.reshape(N, -1) * wt

        out_v, out_a, out_t = [], [], []
        for i in range(B):
            mi = masks[i]  # [3]
            vi, ai, ti = v[i], a[i], t[i]

            q_parts, k_parts = [], []
            if int(mi[0].item()) == 1:
                q_parts.append(vi.reshape(-1) * wv)
                k_parts.append(buf_v_flat)
            if int(mi[1].item()) == 1:
                q_parts.append(ai.reshape(-1) * wa)
                k_parts.append(buf_a_flat)
            if int(mi[2].item()) == 1:
                q_parts.append(ti.reshape(-1) * wt)
                k_parts.append(buf_t_flat)

            if not q_parts:
                out_v.append(vi)
                out_a.append(ai)
                out_t.append(ti)
                continue

            q = torch.cat(q_parts, dim=0).unsqueeze(0)  # [1, Dq]
            Kcat = torch.cat(k_parts, dim=1)  # [N, Dq]
            sims = _cosine_sim(q, Kcat).squeeze(0)  # [N]
            sims = torch.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

            k = min(max(1, self.retrieval_topk), N)
            idx = torch.topk(sims, k=k, largest=True).indices  # [k]

            if int(mi[0].item()) == 0:
                donor_v = buf_v.index_select(0, idx).mean(dim=0)
                vi = _align_temporal(donor_v, vi).to(device)
            if int(mi[1].item()) == 0:
                donor_a = buf_a.index_select(0, idx).mean(dim=0)
                ai = _align_temporal(donor_a, ai).to(device)
            if int(mi[2].item()) == 0:
                donor_t = buf_t.index_select(0, idx).mean(dim=0)
                ti = _align_temporal(donor_t, ti).to(device)

            out_v.append(vi)
            out_a.append(ai)
            out_t.append(ti)

        v_imp = torch.stack(out_v, 0).contiguous()
        a_imp = torch.stack(out_a, 0).contiguous()
        t_imp = torch.stack(out_t, 0).contiguous()
        return v_imp, a_imp, t_imp

    def _get_buffer_minibatch_full(
        self, mb: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        从 buffer 中抽取一个全模态小批，作为生成器训练的额外 GT。
        返回 (v, a, t)（都在 self.device 上）；若 buffer 为空则返回 None。
        """
        if self.buffer.is_empty() or mb <= 0:
            return None
        buf_tuple = self.buffer.get_data(mb, transform=self.transform)[0]
        bv, ba, bt = (x.to(self.device) for x in buf_tuple)
        return (bv, ba, bt)

    # ---------- Generator training (per-batch, learn OFFSETS) ----------
    def _train_generator(
        self,
        gt_full: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        训练生成器学“偏移 Δ”，使 donor + Δ ≈ GT：
          1) 仅模拟 video/audio 缺失（text 不缺）
          2) 用 buffer 全量检索，得到同批 donor (v_ret, a_ret, t_ret)
          3) 前向得到 Δv/Δa/Δt，仅对缺失模态用 L1( donor + Δ , GT )
        """
        v_gt, a_gt, t_gt = (x.to(self.device) for x in gt_full)
        B = v_gt.size(0)
        if B == 0:
            return torch.zeros([], device=self.device)

        # 1) 构造缺失：交替缺 v / 缺 a
        masks_sim = torch.ones((B, 3), dtype=torch.bool, device=self.device)
        miss_idx_va = torch.arange(B, device=self.device) % 2  # 0:video, 1:audio
        masks_sim[torch.arange(B, device=self.device), miss_idx_va] = False

        # 2) 置零后的输入（text 不动）
        v_in, a_in, t_in = v_gt.clone(), a_gt.clone(), t_gt.clone()
        v_in[miss_idx_va == 0] = 0
        a_in[miss_idx_va == 1] = 0

        # 3) 基于 GT 做一次检索（同批 donor）
        with torch.no_grad():
            v_ret, a_ret, t_ret = self._impute_missing_with_buffer(
                (v_gt, a_gt, t_gt), masks_sim
            )

        # 4) 指示器/任务 one-hot（用两类槽位：缺 v / 缺 a）
        miss_ind = masks_sim.float()
        task_1h = torch.zeros((B, self.n_tasks), device=self.device)
        if self.n_tasks > 0:
            valid_idx = miss_idx_va.clamp_max(self.n_tasks - 1)
            task_1h[torch.arange(B, device=self.device), valid_idx] = 1.0

        # 5) 前向预测 Δ
        dv, da, dt = self.generator(
            v_in, a_in, t_in, miss_ind, task_1h, v_ret, a_ret, t_ret
        )

        # 6) donor + Δ 与 GT 的 L1（仅在缺失模态上）
        def masked_l1_missing(pred, donor, tgt, miss_bool_col):
            pred_full = donor + pred
            if pred_full.dim() == 3:
                w = miss_bool_col.view(-1, 1, 1)
            else:
                w = miss_bool_col.view(-1, 1)
            if w.sum() == 0:
                return pred_full.new_tensor(0.0)
            return ((pred_full - tgt).abs() * w).sum() / (w.sum() * pred_full.size(-1))

        # 时间维对齐（谨慎：对 donor/gt 不需要，dv/da/dt 在 forward 已展成相同 T）
        if v_gt.dim() == 3 and dv.size(-2) != v_gt.size(-2):
            dv = _interpolate_T(dv, v_gt.size(-2))
        if a_gt.dim() == 3 and da.size(-2) != a_gt.size(-2):
            da = _interpolate_T(da, a_gt.size(-2))

        loss_v = masked_l1_missing(dv, v_ret, v_gt, miss_idx_va == 0)
        loss_a = masked_l1_missing(da, a_ret, a_gt, miss_idx_va == 1)
        # 不监督 text（稳定）

        return (loss_v + loss_a) * self.gen_rec_weight

    # ---------- Fusion (donor + Δ；text 缺失 -> ret only) ----------
    def _fuse_with_gen_and_ret(
        self, v_inp, a_inp, t_inp, masks, miss_ind, task_1h, v_ret, a_ret, t_ret
    ):
        with torch.no_grad():
            dv, da, dt = self.generator(
                v_inp, a_inp, t_inp, miss_ind, task_1h, v_ret, a_ret, t_ret
            )

            if v_inp.dim() == 3 and dv.size(-2) != v_inp.size(-2):
                dv = _interpolate_T(dv, v_inp.size(-2))
            if a_inp.dim() == 3 and da.size(-2) != a_inp.size(-2):
                da = _interpolate_T(da, a_inp.size(-2))
            if t_inp.dim() == 3 and dt.size(-2) != t_inp.size(-2):
                dt = _interpolate_T(dt, t_inp.size(-2))

            bmv = (
                masks[:, 0].view(-1, 1, 1)
                if v_inp.dim() == 3
                else masks[:, 0].view(-1, 1)
            )
            bma = (
                masks[:, 1].view(-1, 1, 1)
                if a_inp.dim() == 3
                else masks[:, 1].view(-1, 1)
            )
            bmt = masks[:, 2].view(-1, 1, 1)  # text [B,82,512]

            v_fused = torch.where(bmv, v_inp, v_ret + dv)
            a_fused = torch.where(bma, a_inp, a_ret + da)
            t_fused = torch.where(bmt, t_inp, t_ret)  # text 缺失仅 retrieval
        return (v_fused, a_fused, t_fused)

    # ---------- meta batch builder (buffer + current) ----------
    def _build_meta_batches(
        self,
        fused_curr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels_curr: torch.Tensor,
        masks_curr: torch.Tensor,
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        batches = []
        for _ in range(max(1, self.batch_num)):
            if not self.buffer.is_empty():
                mb = getattr(
                    self.args, "minibatch_size", max(1, labels_curr.size(0) // 2)
                )
                buf_tuple = self.buffer.get_data(mb, transform=self.transform)
                (bv, ba, bt), bl = buf_tuple[0], buf_tuple[1]
                bv, ba, bt = bv.to(self.device), ba.to(self.device), bt.to(self.device)
                bl = bl.to(self.device)
                v_mix = torch.cat([bv, fused_curr[0]], dim=0)
                a_mix = torch.cat([ba, fused_curr[1]], dim=0)
                t_mix = torch.cat([bt, fused_curr[2]], dim=0)
                l_mix = torch.cat([bl, labels_curr], dim=0)
                batches.append(((v_mix, a_mix, t_mix), l_mix))
            else:
                batches.append((fused_curr, labels_curr))
        return batches

    # ---------- lifecycle ----------
    def end_task(self, dataset):
        # 保持你之前的 end_task 逻辑（略）
        buf_cap = int(self.args.buffer_size)
        if buf_cap <= 0:
            self.cur_task += 1
            return

        prev_triplet = prev_labels = prev_task_labels = None
        if not self.buffer.is_empty():
            data = self.buffer.get_all_data()
            if isinstance(data, tuple) and len(data) >= 2:
                prev_triplet = data[0]
                prev_labels = data[1]
                if len(data) >= 3:
                    prev_task_labels = data[2]
        self._prev_buffer_cache = (prev_triplet, prev_labels, prev_task_labels)
        self.buffer.empty()

        tasks_seen = self.cur_task + 1
        per_task = max(1, buf_cap // tasks_seen)
        quota_curr = buf_cap - per_task * (tasks_seen - 1)

        # 回灌旧任务
        if (
            hasattr(self, "_prev_buffer_cache")
            and self._prev_buffer_cache[0] is not None
        ):
            pv, pa, pt = self._prev_buffer_cache[0]
            y = self._prev_buffer_cache[1]
            tl = self._prev_buffer_cache[2]
            if pv is not None and y is not None and tl is not None:
                task_ids = tl.long().cpu()
                for tid in torch.unique(task_ids).tolist():
                    mask = task_ids == tid
                    idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if idxs.numel() == 0:
                        continue
                    v_task, a_task, t_task = (
                        pv.index_select(0, idxs),
                        pa.index_select(0, idxs),
                        pt.index_select(0, idxs),
                    )
                    y_task = y.index_select(0, idxs)
                    # 简单：等距采样 per_task（按分数排序版本可按你原函数替换）
                    N = y_task.size(0)
                    take = min(per_task, N)
                    pos = torch.linspace(0, N - 1, steps=take).round().long()
                    sel = pos.clamp_(0, N - 1)
                    v_sel, a_sel, t_sel, y_sel = (
                        v_task.index_select(0, sel),
                        a_task.index_select(0, sel),
                        t_task.index_select(0, sel),
                        y_task.index_select(0, sel),
                    )
                    tl_sel = torch.full((y_sel.size(0),), int(tid), dtype=torch.long)
                    self.buffer.add_data(
                        examples=(v_sel, a_sel, t_sel), labels=y_sel, task_labels=tl_sel
                    )

        # 采样当前任务
        loader = getattr(dataset, "train_loader", None)
        if loader is not None and quota_curr > 0:
            v_list, a_list, t_list, y_list = [], [], [], []
            for batch in loader:
                if isinstance(batch, dict):
                    v, a, t, y = (
                        batch["video"],
                        batch["audio"],
                        batch["text"],
                        batch["labels"],
                    )
                else:
                    (v, a, t), y = batch[0], batch[1]
                v_list.append(v.cpu())
                a_list.append(a.cpu())
                t_list.append(t.cpu())
                y_list.append(y.cpu())
            if len(v_list) > 0:
                v_all, a_all, t_all, y_all = (
                    torch.cat(v_list, 0),
                    torch.cat(a_list, 0),
                    torch.cat(t_list, 0),
                    torch.cat(y_list, 0),
                )
                N = y_all.size(0)
                take = min(quota_curr, N)
                pos = torch.linspace(0, N - 1, steps=take).round().long()
                sel = pos.clamp_(0, N - 1)
                v_sel, a_sel, t_sel, y_sel = (
                    v_all.index_select(0, sel),
                    a_all.index_select(0, sel),
                    t_all.index_select(0, sel),
                    y_all.index_select(0, sel),
                )
                task_labels = torch.full(
                    (y_sel.size(0),), self.cur_task, dtype=torch.long
                )
                self.buffer.add_data(
                    examples=(v_sel, a_sel, t_sel),
                    labels=y_sel,
                    task_labels=task_labels,
                )

        # 打印 buffer 状态
        try:
            data = self.buffer.get_all_data()
            triplet = data[0]
            total = int(triplet[0].size(0))
            if hasattr(self.args, "logging") and self.args.logging is not None:
                self.args.logging.info(f"[Buffer Summary] total = {total}")
                if len(data) >= 3 and isinstance(data[2], torch.Tensor):
                    tl = data[2].detach().cpu().long()
                    for tid in torch.unique(tl).tolist():
                        cnt = int((tl == tid).sum().item())
                        self.args.logging.info(f"    Task {tid+1:02d}: {cnt} samples")
        except Exception:
            pass

        self.cur_task += 1
        if hasattr(self, "_prev_buffer_cache"):
            del self._prev_buffer_cache

    # ---------- training step with meta ----------
    def observe(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        not_aug_inputs: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        epoch: Optional[int] = None,
        task: Optional[int] = None,
    ):
        # unpack
        assert isinstance(inputs, tuple) and len(inputs) == 3
        v_in, a_in, t_in = inputs
        labels = labels.to(self.device)
        B = v_in.size(0)
        masks = _ensure_bool_mask(masks, B, self.device)

        # 1) train generator (offset) with GT
        if not_aug_inputs is not None and not task:
            self.opt_gen.zero_grad(set_to_none=True)
            loss_gen = self._train_generator(not_aug_inputs)
            if torch.isfinite(loss_gen):
                loss_gen.backward()
                self.opt_gen.step()

        # 2) one-shot completion
        fused = self.complete_for_eval((v_in, a_in, t_in), masks, task_id=task)

        # 3) meta: within & across (backbone only)
        theta_A0 = self._snapshot()
        last_loss = None

        meta_batches = self._build_meta_batches(fused, labels, masks)
        for batch_inputs, batch_labels in meta_batches:
            theta_Wi0 = self._snapshot()

            self.opt_backbone.zero_grad(set_to_none=True)
            outputs, _ = self.net(batch_inputs, returnt="all")
            loss_main = self.loss(outputs, batch_labels)
            if not torch.isfinite(loss_main):
                continue
            loss_main.backward()
            self.opt_backbone.step()
            last_loss = loss_main

            if self.beta > 0:
                self._meta_pull(theta_Wi0, self.beta)

        if self.gamma > 0:
            self._meta_pull(theta_A0, self.gamma)

        return float(last_loss.item()) if last_loss is not None else 0.0
