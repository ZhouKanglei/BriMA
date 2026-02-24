#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/11/05
# @Desc: Retrieval-Impute-Replay (RIR): use buffer-similar samples to fill missing modalities and replay.

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer


# ------------------------------ helper functions ------------------------------


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute cosine similarity between [1,D] and [N,D]."""
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a @ b.t()).squeeze(0)


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


# ------------------------------ main model ------------------------------


class RIR(ContinualModel):
    """
    Retrieval-Impute-Replay:
      1. Impute missing modalities using buffer nearest neighbors.
      2. Train on current batch.
      3. Replay from buffer mini-batch for stability.
    """

    NAME = "rir"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(RIR, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, device="cpu")

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # parameters
        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # Retrieval / replay params
        self.retrieval_topk = getattr(self.args, "retrieval_topk", 3)
        self.sim_modal_weights = getattr(
            self.args, "sim_modal_weights", (5.0, 3.0, 1.0)
        )

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

        self.args.logging.info(f"Using {self.batch_num} meta batches per observe.")

        self.beta = getattr(self.args, "beta", 0.25)  # within-batch meta step
        self.gamma = getattr(self.args, "gamma", 0.5)  # across-batch meta step

    # =========================== Internal functions ===========================

    @torch.no_grad()
    def _impute_missing_with_buffer(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fill missing modalities for each sample using nearest samples in buffer.
        inputs: (video, audio, text), each [B, T, C]
        masks:  [B,3], 1=available, 0=missing
        Returns: imputed (video, audio, text)
        """
        v, a, t = inputs
        B = v.size(0)
        device = self.device

        if self.buffer.is_empty():
            return inputs

        buf_ret = self.buffer.get_all_data()
        buf_triplet = buf_ret[0]  # expected ((v,a,t), labels, ...)
        if not (isinstance(buf_triplet, tuple) and len(buf_triplet) == 3):
            return inputs

        buf_v, buf_a, buf_t = (x.to(device) for x in buf_triplet)
        N = buf_v.size(0)
        if N == 0:
            return inputs

        buf_v_flat = buf_v.reshape(N, -1)
        buf_a_flat = buf_a.reshape(N, -1)
        buf_t_flat = buf_t.reshape(N, -1)

        out_v, out_a, out_t = [], [], []
        masks = masks.to(device)
        wv, wa, wt = self.sim_modal_weights

        for i in range(B):
            mi = masks[i]
            vi, ai, ti = v[i], a[i], t[i]

            q_parts, buf_parts = [], []
            if int(mi[0].item()) == 1:
                q_parts.append(vi.reshape(-1) * wv)
                buf_parts.append(buf_v_flat * wv)
            if int(mi[1].item()) == 1:
                q_parts.append(ai.reshape(-1) * wa)
                buf_parts.append(buf_a_flat * wa)
            if int(mi[2].item()) == 1:
                q_parts.append(ti.reshape(-1) * wt)
                buf_parts.append(buf_t_flat * wt)

            if not q_parts:
                out_v.append(vi)
                out_a.append(ai)
                out_t.append(ti)
                continue

            q = torch.cat(q_parts, dim=0).unsqueeze(0)  # [1, Dq]
            Bcat = torch.cat(buf_parts, dim=1)  # [N, Dq]
            sims = _cosine_sim(q, Bcat)  # [N]

            k = min(max(1, self.retrieval_topk), N)
            idx = torch.topk(sims, k=k, largest=True).indices

            if int(mi[0].item()) == 0:
                donor = buf_v[idx].mean(dim=0)
                vi = _align_temporal(donor, vi).to(device)
            if int(mi[1].item()) == 0:
                donor = buf_a[idx].mean(dim=0)
                ai = _align_temporal(donor, ai).to(device)
            if int(mi[2].item()) == 0:
                donor = buf_t[idx].mean(dim=0)
                ti = _align_temporal(donor, ti).to(device)

            out_v.append(vi)
            out_a.append(ai)
            out_t.append(ti)

        return (torch.stack(out_v, 0), torch.stack(out_a, 0), torch.stack(out_t, 0))

    # =========================== Core training function ===========================

    def _ensure_masks(self, masks: torch.Tensor, B: int) -> torch.Tensor:
        """Ensure mask shape [B,3]; if None -> all ones."""
        if masks is None:
            return torch.ones((B, 3), dtype=torch.int, device=self.device)
        if masks.device != self.device:
            masks = masks.to(self.device)
        return masks

    def complete_for_eval(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: torch.Tensor,
        task_id: Optional[int] = None,  # 兼容外部接口；RIR不使用task_id
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一步完成：从“模态缺失的输入” -> “最终可送入backbone的补全模态”
        评测用：仅基于buffer检索进行填补（本RIR不含生成器）。

        用法：
            v_c, a_c, t_c = model.complete_for_eval((v, a, t), masks)
            outputs, _ = model.net((v_c, a_c, t_c), returnt="all")
        """
        assert (
            isinstance(inputs, tuple) and len(inputs) == 3
        ), "inputs must be (video, audio, text)"
        v, a, t = inputs

        # 移到当前设备，确保与buffer/模型一致
        v = v.to(self.device)
        a = a.to(self.device)
        t = t.to(self.device)

        B = v.size(0)
        masks = self._ensure_masks(masks, B)  # -> [B,3] on self.device

        # 仅用buffer进行一次检索式补全（buffer为空则原样返回）
        v_imp, a_imp, t_imp = self._impute_missing_with_buffer((v, a, t), masks)

        # 数值清洗，避免NaN/Inf影响评测稳定性
        def _clean(x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if x.dtype != torch.float32:
                x = x.float()
            return torch.clamp(x, -1e4, 1e4).contiguous()

        v_imp = _clean(v_imp).detach()
        a_imp = _clean(a_imp).detach()
        t_imp = _clean(t_imp).detach()

        return v_imp, a_imp, t_imp

    def observe(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        masks: torch.Tensor = None,
        not_aug_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        epoch=None,
        task=None,
    ):
        """
        Observe one batch:
          1. Impute missing modalities from buffer (internal function)
          2. Train current batch
          3. Draw a mini-batch from buffer for replay
          4. Add current batch into buffer
        """
        v, a, t = inputs
        B = v.size(0)
        masks = self._ensure_masks(masks, B)

        # (1) impute missing modalities internally
        with torch.no_grad():
            inputs_imp = self._impute_missing_with_buffer(inputs, masks)

        # (2) train on current batch
        self.opt.zero_grad()
        outputs = self.net(inputs_imp)
        loss = self.loss(outputs, labels.to(self.device))
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        self.opt.step()

        # (3) buffer replay
        if not self.buffer.is_empty() and self.args.minibatch_size > 0:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )[:2]
            buf_inputs = tuple(x.to(self.device) for x in buf_inputs)
            buf_labels = buf_labels.to(self.device)

            self.opt.zero_grad()
            buf_outputs = self.net(buf_inputs)
            replay_loss = F.mse_loss(buf_outputs["output"], buf_labels)
            total_loss = self.replay_weight * replay_loss
            if torch.isfinite(total_loss):
                total_loss.backward()
                self.opt.step()

        # (4) store current batch into buffer (epoch==0 means first iteration)
        if not epoch:
            to_store = not_aug_inputs if not_aug_inputs is not None else inputs_imp
            to_store = tuple(x.detach().cpu() for x in to_store)
            self.buffer.add_data(examples=to_store, labels=labels.detach().cpu())

        return float(loss.item())
