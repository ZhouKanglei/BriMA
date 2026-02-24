#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/18 下午5:39

import torch
import torch.nn.functional as F
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer
from utils.metrics import DRLoss


# ----------------------------- helpers -----------------------------


def generate_random_orthogonal_matrix(feat_in: int, num_classes: int) -> torch.Tensor:
    """
    Generate a (feat_in x num_classes) orthonormal column matrix via QR.
    Columns are orthonormal: Q^T Q = I.
    """
    rand_mat = np.random.randn(feat_in, num_classes)
    q, _ = np.linalg.qr(rand_mat)
    q = torch.tensor(q, dtype=torch.float32)
    eye = torch.eye(num_classes, dtype=torch.float32)
    assert torch.allclose(
        q.T @ q, eye, atol=1e-5
    ), f"Orthonormal check failed, max err={torch.max(torch.abs(q.T @ q - eye)).item():.2e}"
    return q


def _to_device_inputs(inp_tuple, device):
    """Move a (video, audio, text) tuple to device."""
    return tuple(x.to(device) for x in inp_tuple)


def _slice_inputs(inp_tuple, idx):
    """Slice a (video, audio, text) tuple by given indices tensor/list."""
    return tuple(x[idx] for x in inp_tuple)


def _flatten_and_concat_features(features) -> torch.Tensor:
    """
    Accept features in several formats and return a 2D tensor [B, F]:
      - tuple/list: (f_v, f_a, f_t), each [B, ...]
      - dict: {"video": ..., "audio": ..., "text": ...}
      - tensor: [B, ...] (legacy)
    """
    if isinstance(features, (tuple, list)):
        flats = [f.reshape(f.shape[0], -1) for f in features]
        return torch.cat(flats, dim=1)
    if isinstance(features, dict):
        parts = []
        for key in ("video", "audio", "text"):
            if key in features and features[key] is not None:
                f = features[key]
                parts.append(f.reshape(f.shape[0], -1))
        if len(parts) == 0:
            return None
        return torch.cat(parts, dim=1)
    if torch.is_tensor(features):
        return features.reshape(features.shape[0], -1)
    return None


# ----------------------------- Nc -----------------------------


class Nc(ContinualModel):
    NAME = "nc"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Nc, self).__init__(backbone, loss, args, transform)

        # multimodal buffer on CPU (you can set to cuda if desired)
        self.buffer = Buffer(self.args.buffer_size, device="cpu")
        self.buffer.empty()

        # DR targets: lazy-init once we know total feature dim
        self.orth_vec = None
        self.num_classes = getattr(args, "num_classes", 5)

        self.dr_loss = DRLoss()

        self.current_task = 0
        self.lambda1 = args.alpha
        self.n_tasks = (
            args.n_tasks + 1 if getattr(args, "fewshot", False) else args.n_tasks
        )

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    # ---------------------- sample selection (simplified) ----------------------

    @staticmethod
    def _evenly_spaced_indices(n: int, k: int):
        """Return k indices evenly spaced in [0, n-1] (no replacement)."""
        if k >= n:
            return list(range(n))
        # 用 linspace 再去重/求整
        idx = torch.linspace(0, n - 1, steps=k).round().long().tolist()
        # 去重保持顺序
        seen, out = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                out.append(i)
        # 如果去重后不够，补充
        ptr = 0
        while len(out) < k:
            if ptr not in seen:
                out.append(ptr)
                seen.add(ptr)
            ptr += 1
        return out

    def select_examples(self, labels_1d: torch.Tensor, num: int):
        """
        Rank-based evenly spaced selection.
        labels_1d: tensor [N] (float)
        """
        N = labels_1d.numel()
        if num >= N:
            return list(range(N))
        order = torch.argsort(labels_1d)  # ascending
        picks_rank = self._evenly_spaced_indices(N, num)
        return order[picks_rank].tolist()

    # ---------------------- buffer population per task ----------------------

    def data2buffer_ous(self, dataset):
        """
        Select a small set from current task and push into buffer (multimodal).
        本地 DataLoader 两遍：
          1) 收集所有 label，决定全局要挑选的 indices（按 rank 等距）
          2) 再遍历一次，根据这些全局 index 把对应样本写入 buffer
        """
        examples_per_task = self.args.buffer_size // max(1, (self.n_tasks - 1))
        self.args.logging.info(
            f"Current task {self.current_task} - select {examples_per_task} samples"
        )

        # pass 1: collect all labels
        all_labels = []
        for batch in dataset.train_loader:
            if isinstance(batch, dict):
                labels = batch["labels"]
            else:
                _, labels, _ = batch
            all_labels.append(labels.reshape(-1).cpu())
        all_labels = torch.cat(all_labels, dim=0)  # [N]

        if all_labels.numel() == 0:
            return

        global_sel_idx = self.select_examples(all_labels, examples_per_task)
        select_mask = torch.zeros_like(all_labels, dtype=torch.bool)
        select_mask[torch.tensor(global_sel_idx, dtype=torch.long)] = True

        # pass 2: walk again and push selected rows
        cursor = 0
        with torch.no_grad():
            for batch in dataset.train_loader:
                if isinstance(batch, dict):
                    inputs = (batch["video"], batch["audio"], batch["text"])
                    labels = batch["labels"]
                else:
                    inputs, labels, _ = batch

                B = labels.shape[0]
                local_mask = select_mask[cursor : cursor + B]
                cursor += B
                if not local_mask.any():
                    continue

                idx = torch.nonzero(local_mask, as_tuple=False).squeeze(1)
                inputs_sel = _slice_inputs(inputs, idx)
                labels_sel = labels[idx].detach().cpu()

                self.buffer.add_data(
                    examples=tuple(x.detach().cpu() for x in inputs_sel),
                    labels=labels_sel,
                    task_labels=torch.full(
                        (labels_sel.shape[0],), self.current_task, dtype=torch.long
                    ),
                )

        # stats
        try:
            all_ret = self.buffer.get_all_data()
            # ((v,a,t), labels, [logits]?, [task_labels]?, [masks]?)
            task_labels = None
            if (
                len(all_ret) >= 3
                and all_ret[2] is not None
                and all_ret[2].dtype == torch.long
            ):
                task_labels = all_ret[2]
            elif (
                len(all_ret) >= 4
                and all_ret[3] is not None
                and all_ret[3].dtype == torch.long
            ):
                task_labels = all_ret[3]
            if task_labels is not None:
                for ttl in task_labels.unique():
                    cnt = int((task_labels == ttl).sum().item())
                    self.args.logging.info(
                        f"Task {int(ttl)} has {cnt} samples in the buffer."
                    )
        except Exception:
            pass

    def end_task(self, dataset):
        self.current_task += 1
        if self.current_task < self.n_tasks:
            self.data2buffer_ous(dataset)

    # ---------------------- training step ----------------------

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        """
        inputs: (video, audio, text)
        net(..., returnt="all") should return:
          outputs: dict with key 'output'
          features: could be tuple/dict/tensor (we will concat them to [B,F])
        """
        self.opt.zero_grad()

        # forward current batch
        outputs, features = self.net(inputs, returnt="all")
        preds = outputs["output"] if isinstance(outputs, dict) else outputs
        loss = self.loss(outputs, labels)  # 你的损失函数从 outputs['output'] 读取

        # init orth_vec lazily using total feature dim (concat of modalities)
        feat_flat = _flatten_and_concat_features(features)
        if (self.orth_vec is None) and (feat_flat is not None):
            feat_flat_dim = feat_flat.shape[1]
            self.orth_vec = generate_random_orthogonal_matrix(
                feat_in=feat_flat_dim, num_classes=self.num_classes
            ).to(self.device)

        # replay
        if not self.buffer.is_empty():
            ((v_b, a_b, t_b), buf_labels, *_) = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_inputs = (v_b.to(self.device), a_b.to(self.device), t_b.to(self.device))
            buf_labels = buf_labels.to(self.device)

            buf_outputs, buf_features = self.net(buf_inputs, returnt="all")
            buf_preds = (
                buf_outputs["output"] if isinstance(buf_outputs, dict) else buf_outputs
            )

            # supervised replay loss
            loss += F.mse_loss(buf_preds, buf_labels)

            # DR regularization (only if we have features + basis)
            buf_feat_flat = _flatten_and_concat_features(buf_features)
            if (
                (feat_flat is not None)
                and (buf_feat_flat is not None)
                and (self.orth_vec is not None)
            ):
                joint_features = torch.cat(
                    [buf_feat_flat, feat_flat], dim=0
                )  # [B_buf+B_cur, F]
                joint_labels = torch.cat(
                    [buf_labels, labels.to(self.device)], dim=0
                ).reshape(-1)

                # 如果你的 label 是从 1 开始，可用 -1；否则直接 clamp 到 [0, num_classes-1]
                cls_idx = joint_labels.long().clamp(min=0, max=self.num_classes - 1)

                # 目标“原型” [F, C] -> 取出每个样本的列 -> [F, B] -> 转置成 [B, F]
                tgt = self.orth_vec[:, cls_idx].T.contiguous()
                loss += self.lambda1 * self.dr_loss(tgt, joint_features)

        # step
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        # 写入 buffer（只在首个 epoch；可按你需要改策略）
        if epoch == 0 and not_aug_inputs is not None:
            self.buffer.add_data(
                examples=tuple(x.detach().cpu() for x in not_aug_inputs),
                labels=labels.detach().cpu(),
            )

        return loss.item()
