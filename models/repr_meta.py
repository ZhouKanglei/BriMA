#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/11/06
# @Desc: RePRMeta — Retrieval + Prompted Generation with Reptile-style Meta
#       - inputs: (video, audio, text[B,82,512])
#       - generator trained per-batch with GT (构造缺失：i%3)
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
# Prompted Generator
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
    """Prompted generator: [p_missing, p_task, v_tok, a_tok, t_tok] -> Transformer -> heads."""

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
        self.p_m = _PromptMLP(3, hid)
        self.p_t = _PromptMLP(n_tasks, hid)
        enc = nn.TransformerEncoderLayer(
            d_model=hid, nhead=nhead, dim_feedforward=hid * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.v_head = nn.Linear(hid, v_dim)
        self.a_head = nn.Linear(hid, a_dim)
        self.t_head = nn.Linear(hid, t_dim)

    def forward(self, v, a, t, miss_ind, task_1h):
        vt, at, tt = (
            self.v_proj(_flatten_time(v)),
            self.a_proj(_flatten_time(a)),
            self.t_proj(_flatten_time(t)),
        )
        seq = torch.stack(
            [self.p_m(miss_ind), self.p_t(task_1h), vt, at, tt], dim=1
        )  # [B,5,hid]
        h = self.encoder(seq)
        v_gen, a_gen, t_gen = (
            self.v_head(h[:, 2, :]),
            self.a_head(h[:, 3, :]),
            self.t_head(h[:, 4, :]),
        )
        if v.dim() == 3:
            v_gen = v_gen.unsqueeze(1).expand(-1, v.size(-2), -1).contiguous()
        if a.dim() == 3:
            a_gen = a_gen.unsqueeze(1).expand(-1, a.size(-2), -1).contiguous()
        if t.dim() == 3:
            t_gen = t_gen.unsqueeze(1).expand(-1, t.size(-2), -1).contiguous()
        return v_gen, a_gen, t_gen


# ----------------------------- #
# RePRMeta (with Reptile-style Meta)
# ----------------------------- #
class RePRMeta(ContinualModel):
    """
    Retrieval + Prompted Generation + Reptile-style Meta.

    - Input:
        inputs: (video, audio, text[B,82,512])
        masks:  [B,3] 1=present, 0=missing (missing parts of inputs have been set to zero)
    - Strategy:
        If text is missing -> only retrieval
        If video/audio is missing -> generator + λ·retrieval
    - Evaluation:
        v_c, a_c, t_c = self.complete_for_eval((v, a, t), masks, task_id)
        outputs, _ = self.net((v_c, a_c, t_c), returnt="all")
    """

    NAME = "repr_meta"
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

        self.ret_lambda = getattr(self.args, "ret_lambda", 0.75)

        # Meta params
        batch_num = 1
        if not hasattr(self.args, "batch_num_non_default"):
            if self.args.dataset.endswith("rg"):
                batch_num = 3
            if self.args.dataset.endswith("fs"):
                batch_num = 1
            if self.args.dataset.endswith("fs1000"):
                batch_num = 3
            self.batch_num = batch_num  # number of inner batches per observe
            print(self.args.dataset, "=> setting batch_num =", self.batch_num)
        else:
            self.batch_num = self.args.batch_num

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
            self.args.logging.info(f"[RePRMeta] Generator params: {total_params}")

            # retrieval weights
            self.args.logging.info(
                f"[RePRMeta] Retrieval sim weights: video={self.sim_modal_weights[0]}, "
                f"audio={self.sim_modal_weights[1]}, text={self.sim_modal_weights[2]}"
            )
            self.args.logging.info(
                f"[RePRMeta] Retrieval top-k: {self.retrieval_topk}, "
                f"fusion λ: {self.ret_lambda}"
            )
            # fusion weights
            self.args.logging.info(f"[RePRMeta] Retrieval weight: {self.ret_lambda}")

    # ---------- helpers ----------
    def _indicators(self, masks: torch.Tensor, task_id: Optional[int] = None):
        B = masks.size(0)
        miss_ind = masks.float().to(self.device)
        # 若传入 task_id，用它；否则用 self.cur_task
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

        # 先检索
        v_ret, a_ret, t_ret = self._impute_missing_with_buffer((v, a, t), masks)

        # 再融合（生成器只取值），这里用传入的 task_id 影响 prompt one-hot
        with torch.no_grad():
            miss_ind, task_1h = self._indicators(masks, task_id=task_id)
            gen_v, gen_a, gen_t = self.generator(v, a, t, miss_ind, task_1h)

            if v.dim() == 3 and gen_v.size(-2) != v.size(-2):
                gen_v = _interpolate_T(gen_v, v.size(-2))
            if a.dim() == 3 and gen_a.size(-2) != a.size(-2):
                gen_a = _interpolate_T(gen_a, a.size(-2))
            if t.dim() == 3 and gen_t.size(-2) != t.size(-2):
                gen_t = _interpolate_T(gen_t, t.size(-2))

            bmv = (
                masks[:, 0].view(-1, 1, 1) if v.dim() == 3 else masks[:, 0].view(-1, 1)
            )
            bma = (
                masks[:, 1].view(-1, 1, 1) if a.dim() == 3 else masks[:, 1].view(-1, 1)
            )
            bmt = masks[:, 2].view(-1, 1, 1)  # text [B,82,512]

            v_fused = torch.where(
                bmv, v, gen_v * (1 - self.ret_lambda) + self.ret_lambda * v_ret
            )
            a_fused = torch.where(
                bma, a, gen_a * (1 - self.ret_lambda) + self.ret_lambda * a_ret
            )
            t_fused = torch.where(bmt, t, t_ret)  # text缺失仅 retrieval

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

        # 取 buffer 全量数据：期望 ((v,a,t), labels, task_labels)
        buf_all = self.buffer.get_all_data()
        buf_triplet = buf_all[0]
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
        buf_tuple = self.buffer.get_data(mb, transform=self.transform)[
            0
        ]  # ((v,a,t), labels, [task_labels])
        bv, ba, bt = (x.to(self.device) for x in buf_tuple)
        return (bv, ba, bt)

    # ---------- Generator training (per-batch) ----------
    def _train_generator(
        self,
        gt_full: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        仅用 GT（not_aug_inputs 或/及 buffer 拼接）训练生成器：
        - 只模拟 video/audio 缺失（不再让 text 缺失）
        - 仅对缺失的那一模态计算重构损失
        """
        v_gt, a_gt, t_gt = (x.to(self.device) for x in gt_full)
        B = v_gt.size(0)
        if B == 0:
            return torch.zeros([], device=self.device)

        # 1) 只在 {video, audio} 里制造缺失：0->video, 1->audio
        masks_sim = torch.ones((B, 3), dtype=torch.bool, device=self.device)
        miss_idx_va = torch.arange(B, device=self.device) % 2  # 0/1 交替
        # 对应到三模态的列：0=video, 1=audio, 2=text（不缺）
        masks_sim[torch.arange(B, device=self.device), miss_idx_va] = False

        # 2) 构造缺失后的输入（置零），注意 text 不动
        v_in, a_in, t_in = v_gt.clone(), a_gt.clone(), t_gt.clone()
        v_in[miss_idx_va == 0] = 0  # 缺 video 的样本
        a_in[miss_idx_va == 1] = 0  # 缺 audio 的样本
        # t_in 保持原样（text 不参与缺失）

        # 3) 指示器由 mask 决定；task one-hot 也只用两类（缺 v / 缺 a）
        miss_ind = masks_sim.float()  # [B,3]
        task_1h = torch.zeros((B, self.n_tasks), device=self.device)
        if self.n_tasks > 0:
            # 用 0/1 两个槽位表示“缺 video / 缺 audio”
            valid_idx = miss_idx_va.clamp_max(self.n_tasks - 1)
            task_1h[torch.arange(B, device=self.device), valid_idx] = 1.0

        # 4) 前向生成
        v_gen, a_gen, t_gen = self.generator(v_in, a_in, t_in, miss_ind, task_1h)

        # 5) 时间对齐
        if v_gt.dim() == 3 and v_gen.size(-2) != v_gt.size(-2):
            v_gen = _interpolate_T(v_gen, v_gt.size(-2))
        if a_gt.dim() == 3 and a_gen.size(-2) != a_gt.size(-2):
            a_gen = _interpolate_T(a_gen, a_gt.size(-2))
        # t_gen 不用于监督，无需对齐（保持一致也无妨）
        if t_gt.dim() == 3 and t_gen.size(-2) != t_gt.size(-2):
            t_gen = _interpolate_T(t_gen, t_gt.size(-2))

        # 6) 仅对缺失模态计算重构损失（不对 text 计算）
        def masked_l1_missing(pred, tgt, miss_bool_col):
            if pred.dim() == 3:
                w = miss_bool_col.view(-1, 1, 1)
            else:
                w = miss_bool_col.view(-1, 1)
            if w.sum() == 0:
                return pred.new_tensor(0.0)
            return ((pred - tgt).abs() * w).sum() / (w.sum() * pred.size(-1))

        loss_v = masked_l1_missing(
            v_gen, v_gt, miss_idx_va == 0
        )  # 只在缺 v 的样本上监督 v
        loss_a = masked_l1_missing(
            a_gen, a_gt, miss_idx_va == 1
        )  # 只在缺 a 的样本上监督 a
        # 不对 text 计算损失

        return (loss_v + loss_a) * self.gen_rec_weight

    # ---------- Fusion (gen + λ·ret; text 缺失 -> ret only) ----------
    def _fuse_with_gen_and_ret(
        self, v_inp, a_inp, t_inp, masks, miss_ind, task_1h, v_ret, a_ret, t_ret
    ):
        with torch.no_grad():
            gen_v, gen_a, gen_t = self.generator(v_inp, a_inp, t_inp, miss_ind, task_1h)

            if v_inp.dim() == 3 and gen_v.size(-2) != v_inp.size(-2):
                gen_v = _interpolate_T(gen_v, v_inp.size(-2))
            if a_inp.dim() == 3 and gen_a.size(-2) != a_inp.size(-2):
                gen_a = _interpolate_T(gen_a, a_inp.size(-2))
            if t_inp.dim() == 3 and gen_t.size(-2) != t_inp.size(-2):
                gen_t = _interpolate_T(gen_t, t_inp.size(-2))

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

            v_fused = torch.where(
                bmv, v_inp, gen_v * (1 - self.ret_lambda) + self.ret_lambda * v_ret
            )
            a_fused = torch.where(
                bma, a_inp, gen_a * (1 - self.ret_lambda) + self.ret_lambda * a_ret
            )
            t_fused = torch.where(bmt, t_inp, t_ret)  # text 缺失仅 retrieval

        return (v_fused, a_fused, t_fused)

    # ---------- meta batch builder (buffer + current) ----------
    def _build_meta_batches(
        self,
        fused_curr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels_curr: torch.Tensor,
        masks_curr: torch.Tensor,
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        构造 batch_num 个 meta 批次：每个批次 = concat(buffer_minibatch, current_fused)
        buffer 样本默认视为全模态可用（不需要 masks）
        """
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

    # =========================
    # Buffer selection helpers (no mask, no logits)
    # =========================

    def _gather_loader_to_cpu(self, loader):
        if loader is None:
            return None
        v_list, a_list, t_list, y_list = [], [], [], []
        for batch in loader:
            if isinstance(batch, dict):
                v, a, t = batch["video"], batch["audio"], batch["text"]
                y = batch["labels"]
            else:
                (v, a, t), y = batch[0], batch[1]
            v_list.append(v.cpu())
            a_list.append(a.cpu())
            t_list.append(t.cpu())
            y_list.append(y.cpu())
        if len(v_list) == 0:
            return None
        return (
            torch.cat(v_list, dim=0),
            torch.cat(a_list, dim=0),
            torch.cat(t_list, dim=0),
            torch.cat(y_list, dim=0),
        )

    def _sort_and_stride_sample(
        self, v_all, a_all, t_all, y_all, take, descending: bool = False
    ):
        N = y_all.size(0)
        if N == 0 or take <= 0:
            return None

        if y_all.dim() == 1:
            score = y_all
        else:
            score = y_all.view(N, -1).mean(dim=-1)

        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        sort_idx = torch.argsort(score, descending=descending, stable=True)

        take = min(take, N)
        positions = torch.linspace(0, N - 1, steps=take, device=sort_idx.device)
        positions = positions.round().long().clamp_(0, N - 1)
        positions = torch.unique(positions, sorted=True)

        sel_idx = sort_idx.index_select(0, positions)

        return (
            v_all.index_select(0, sel_idx),
            a_all.index_select(0, sel_idx),
            t_all.index_select(0, sel_idx),
            y_all.index_select(0, sel_idx),
        )

    def _sample_prev_from_buffer(self, per_task: int):
        if not hasattr(self, "_prev_buffer_cache"):
            return
        prev_triplet, prev_labels, prev_task_labels = self._prev_buffer_cache
        if prev_triplet is None or prev_labels is None or prev_task_labels is None:
            return
        pv, pa, pt = prev_triplet
        task_ids = prev_task_labels.long().cpu()
        unique_tasks = torch.unique(task_ids)
        for tid in unique_tasks.tolist():
            mask = task_ids == tid
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue
            v_task = pv.index_select(0, idxs)
            a_task = pa.index_select(0, idxs)
            t_task = pt.index_select(0, idxs)
            y_task = prev_labels.index_select(0, idxs)
            sel = self._sort_and_stride_sample(v_task, a_task, t_task, y_task, per_task)
            if sel is None:
                continue
            v_sel, a_sel, t_sel, y_sel = sel
            task_labels_sel = torch.full((y_sel.size(0),), int(tid), dtype=torch.long)
            self.buffer.add_data(
                examples=(v_sel, a_sel, t_sel),
                labels=y_sel,
                task_labels=task_labels_sel,
            )

    def _print_buffer_status(self):
        """
        Robust buffer summary printer (no buffer modification).

        期望 get_all_data() 返回至少:
            ((v, a, t), labels) 或 ((v, a, t), labels, task_labels)
        - 自动判断 task_labels 是否存在，若不存在则只打印 total。
        - 任务编号用 1-based (Task 01, Task 02...)。
        """
        log = getattr(self.args, "logging", None)

        def _log(msg):
            if log is not None:
                log.info(msg)
            else:
                print(msg)

        # 若为空，get_all_data 会抛错，这里包一层
        try:
            data = self.buffer.get_all_data()
        except Exception:
            _log("[Buffer Summary] total = 0")
            return

        if not isinstance(data, tuple) or len(data) < 2:
            _log("[Buffer Summary] buffer structure invalid.")
            return

        triplet = data[0]  # (v, a, t)
        labels = data[1]  # labels tensor or None
        task_labels = None

        # 推断总样本数
        if (
            isinstance(triplet, tuple)
            and len(triplet) == 3
            and isinstance(triplet[0], torch.Tensor)
        ):
            total = int(triplet[0].size(0))
        elif isinstance(labels, torch.Tensor):
            total = int(labels.size(0))
        else:
            total = 0

        _log(f"[Buffer Summary] total = {total}")

        # 捕获可选的 task_labels：第三项（若存在且形状合理）
        if len(data) >= 3 and isinstance(data[2], torch.Tensor):
            tl = data[2]
            # 要求一维整型、长度与 labels 对齐（若 labels 存在）
            if tl.dim() == 1 and tl.dtype in (
                torch.int64,
                torch.int32,
                torch.long,
                torch.int,
            ):
                if (labels is None) or (tl.numel() == labels.numel()):
                    task_labels = tl

        # 如果没有 task_labels 就不分任务打印
        if task_labels is None:
            return

        tl = task_labels.detach().cpu().long()
        if tl.numel() == 0:
            return

        uniq = torch.unique(tl)
        # 排序并逐任务统计
        for tid0 in uniq.tolist():
            cnt = int((tl == tid0).sum().item())
            _log(f"    Task {tid0 + 1:02d}: {cnt} samples")

    # ---------- lifecycle ----------
    def end_task(self, dataset):
        # >>>>>> 你已有的 end_task 逻辑保持不变 <<<<<<
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

        self._sample_prev_from_buffer(per_task)

        loader = getattr(dataset, "train_loader", None)
        gathered = self._gather_loader_to_cpu(loader)
        if gathered is not None and quota_curr > 0:
            v_all, a_all, t_all, y_all = gathered
            sel = self._sort_and_stride_sample(v_all, a_all, t_all, y_all, quota_curr)
            if sel is not None:
                v_sel, a_sel, t_sel, y_sel = sel
                task_labels = torch.full(
                    (y_sel.size(0),), self.cur_task, dtype=torch.long
                )
                self.buffer.add_data(
                    examples=(v_sel, a_sel, t_sel),
                    labels=y_sel,
                    task_labels=task_labels,
                )

        self._print_buffer_status()
        self.cur_task += 1
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
        # 0) unpack
        assert isinstance(inputs, tuple) and len(inputs) == 3
        v_in, a_in, t_in = inputs
        labels = labels.to(self.device)
        B = v_in.size(0)
        masks = _ensure_bool_mask(masks, B, self.device)

        # 1) 训练生成器（仅用 not_aug_inputs / GT；不参与 meta，不更新 backbone）
        if not_aug_inputs is not None and not task:
            loss_gen = self._train_generator(not_aug_inputs)
            if torch.isfinite(loss_gen):
                self.opt_gen.zero_grad(set_to_none=True)
                loss_gen.backward()
                self.opt_gen.step()

        # 2) 一次性补全（检索 + 生成融合），可传入外部 task id
        fused = self.complete_for_eval((v_in, a_in, t_in), masks, task_id=task)

        # 3) Meta — within & across，仅更新 backbone
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
