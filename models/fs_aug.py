# -*- coding: utf-8 -*-
# @Time: 2023/7/23 23:33

import torch
import torch.nn.functional as F
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer  # ★ 改为多模态 Buffer


# ------------- helpers: 多模态特征打包/解包 -------------


def _pack_features(feats):
    """
    将多模态特征打包为 [B, F]，同时记录每个模态的原始 shape 以便还原。
    支持:
      - tuple/list: (f_v, f_a, f_t)
      - dict: {'video':..., 'audio':..., 'text':...}
      - tensor: 单模态
    返回:
      flat [B,F], meta=[{'name':str,'shape':tuple}], order=list[str]
    """
    parts, meta, order = [], [], []

    def _append(name, t):
        parts.append(t.reshape(t.shape[0], -1))
        meta.append({"name": name, "shape": tuple(t.shape)})
        order.append(name)

    if isinstance(feats, (tuple, list)):
        names = ["video", "audio", "text"]
        for i, t in enumerate(feats):
            if t is None:
                continue
            nm = names[i] if i < len(names) else f"m{i}"
            _append(nm, t)
    elif isinstance(feats, dict):
        for nm in ("video", "audio", "text"):
            if nm in feats and feats[nm] is not None:
                _append(nm, feats[nm])
    elif torch.is_tensor(feats):
        _append("feat", feats)
    else:
        raise TypeError(f"Unsupported features type: {type(feats)}")

    if len(parts) == 0:
        raise ValueError("No valid modality features to pack.")

    flat = torch.cat(parts, dim=1)
    return flat, meta, order


def _unpack_features(flat, meta, order):
    """
    按 meta 与 order 将 [B,F] 还原为 (video, audio, text) 的 tuple（或单模态）。
    要求 flat 的列数等于各模态 numel 之和。
    """
    B = flat.shape[0]
    outs_by_name = {}
    offset = 0
    for m in meta:
        shape = m["shape"]  # (B, ...)
        numel = 1
        for s in shape[1:]:
            numel *= s
        chunk = flat[:, offset : offset + numel]
        offset += numel
        t = chunk.reshape((B, *shape[1:]))
        outs_by_name[m["name"]] = t
    out_tuple = tuple(outs_by_name[nm] for nm in order if nm in outs_by_name)
    return out_tuple


class FsAug(ContinualModel):
    NAME = "fs_aug"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(FsAug, self).__init__(backbone, loss, args, transform)

        # 多模态缓冲区：存 (video, audio, text) + labels
        self.buffer = Buffer(self.args.buffer_size, "cpu")

        self.current_task = 0
        self.i = 0
        self.n_tasks = args.n_tasks

        # 若 backbone 是固定的，可只优化 regressor；否则保留 feature_extractor
        self.opt = torch.optim.Adam(
            params=[
                {"params": self.net.feature_extractor.parameters(), "lr": self.args.lr},
                {"params": self.net.regressor.parameters(), "lr": self.args.lr},
            ],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # set seed
        np.random.seed(args.seed)

    # ---------------- 多模态输入写入 buffer（按 rank 均匀抽样） ----------------

    @staticmethod
    def _evenly_spaced_indices(n: int, k: int):
        if k >= n:
            return list(range(n))
        idx = torch.linspace(0, n - 1, steps=k).round().long().tolist()
        seen, out = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                out.append(i)
        ptr = 0
        while len(out) < k:
            if ptr not in seen:
                out.append(ptr)
                seen.add(ptr)
            ptr += 1
        return out

    @staticmethod
    def _select_by_rank_even(labels_1d: torch.Tensor, k: int):
        n = labels_1d.numel()
        if k >= n:
            return list(range(n))
        order = torch.argsort(labels_1d)  # ascend
        picks = FsAug._evenly_spaced_indices(n, k)
        return order[picks].tolist()

    def inputs2buffer(self, dataset):
        examples_per_task = max(1, self.args.buffer_size // max(1, (self.n_tasks - 1)))
        self.args.logging.info(
            f"Current task {self.current_task} - select {examples_per_task}"
        )

        # Pass1: 收集所有标签
        all_labels = []
        for batch in dataset.train_loader:
            if isinstance(batch, dict):
                labels = batch["labels"]
            else:
                _, labels, _ = batch
            all_labels.append(labels.reshape(-1).cpu())
        all_labels = torch.cat(all_labels, dim=0)

        if all_labels.numel() == 0:
            return

        global_idx = self._select_by_rank_even(all_labels, examples_per_task)
        select_mask = torch.zeros_like(all_labels, dtype=torch.bool)
        select_mask[torch.tensor(global_idx, dtype=torch.long)] = True

        # Pass2: 写入 buffer
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
                sel_inputs = tuple(x[idx].detach().cpu() for x in inputs)
                sel_labels = labels[idx].detach().cpu()

                self.buffer.add_data(
                    examples=sel_inputs,
                    labels=sel_labels,
                    task_labels=torch.full(
                        (sel_labels.shape[0],), self.current_task, dtype=torch.long
                    ),
                )

        # 统计
        try:
            all_ret = self.buffer.get_all_data()
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
            self.inputs2buffer(dataset)

    # ---------------- 主训练 + 特征-分数联合增强 ----------------

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        self.i += 1

        # ------- 主分支 -------
        self.opt.zero_grad()
        outputs, features = self.net(
            inputs, returnt="all"
        )  # outputs: dict, features: tuple/dict/tensor
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # ------- 增强分支（回放 + helper 全量） -------
        if not self.buffer.is_empty():
            # 取回放 batch
            ((v_b, a_b, t_b), buf_labels, *_) = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_inputs = (v_b.to(self.device), a_b.to(self.device), t_b.to(self.device))
            buf_labels = buf_labels.to(self.device)

            # 取全部 helper（可能较大；如需可随机子采样）
            ((v_all, a_all, t_all), lab_all, *_) = self.buffer.get_all_data()
            helper_inputs = (
                v_all.to(self.device),
                a_all.to(self.device),
                t_all.to(self.device),
            )
            helper_labels = lab_all.to(self.device)

            # 前向得到特征（只用特征做增强）
            self.opt.zero_grad()
            with torch.no_grad():
                _, buf_feats = self.net(buf_inputs, returnt="all")
                _, helper_feats = self.net(helper_inputs, returnt="all")

            buf_flat, buf_meta, buf_order = _pack_features(buf_feats)
            helper_flat, _, _ = _pack_features(helper_feats)

            # 做特征-分数联合增强（在 flat 空间上）
            aug_flat, aug_scores = self.feat_score_aug(
                feature1=buf_flat,  # [B,F]
                feature_list=helper_flat,  # [N,F]（会按 batch broadcast 运算）
                score1=buf_labels,  # [B] 或 [B,1...]
                score_list=helper_labels,  # [N] 或 [N,1...]
                aug_scale=0.3,
                count=self.i,
            )

            # 按原模态形状解包成 (video, audio, text) 特征，再回归增强分数
            aug_tuple = _unpack_features(aug_flat, buf_meta, buf_order)
            aug_pred = self.net.regressor(aug_tuple)
            if isinstance(aug_pred, dict) and "output" in aug_pred:
                aug_pred = aug_pred["output"]

            # 形状对齐
            tgt = aug_scores.float().to(aug_pred.device)
            if aug_pred.dim() == 2 and aug_pred.size(1) == 1 and tgt.dim() == 1:
                tgt = tgt.unsqueeze(1)
            elif aug_pred.dim() == 1 and tgt.dim() == 2 and tgt.size(1) == 1:
                aug_pred = aug_pred.unsqueeze(1)

            loss_aug = F.mse_loss(aug_pred, tgt)
            assert torch.isfinite(loss_aug).all(), "NaN/Inf in augmentation loss"

            loss_aug.backward()
            self.opt.step()

        # （可选）在线把未增强样本写入 buffer：此处 FsAug 不写，保持 selection-only
        # if epoch == 0 and not_aug_inputs is not None:
        #     self.buffer.add_data(
        #         examples=tuple(x.detach().cpu() for x in not_aug_inputs),
        #         labels=labels.detach().cpu()
        #     )

        return loss.item()

    # ---------------- 特征-分数联合增强（在 flat 特征空间） ----------------

    def feat_score_aug(
        self, feature1, feature_list, score1, score_list, aug_scale=0.3, count=None
    ):
        """
        feature1:  [B, F]
        feature_list: [N, F]  （helper pool）
        score1:    [B] / [B,1] / [B,K]
        score_list:[N] / [N,1] / [N,K]

        增强策略：累加 helper 与当前 batch 的差值，缩放后施加随机幅度 r ~ N(0, aug_scale)。
        """
        # 广播对齐到 [N, B, F] / [N, B, ...]
        B, F = feature1.shape
        N = feature_list.shape[0]

        # [N, B, F]
        feat1_rep = feature1.unsqueeze(0).expand(N, B, F)
        feat_list_rep = feature_list.unsqueeze(1).expand(N, B, F)
        feat_diff = feat_list_rep - feat1_rep  # [N,B,F]

        # 分数对齐
        if score1.dim() == 1:
            s1 = score1.view(B, 1)  # [B,1]
        else:
            s1 = score1  # [B,K]
        if score_list.dim() == 1:
            sL = score_list.view(N, 1)  # [N,1]
        else:
            sL = score_list  # [N,K]
        s1_rep = s1.unsqueeze(0).expand(N, *s1.shape)  # [N,B,K] or [N,B,1]
        sL_rep = sL.unsqueeze(1).expand(N, *s1.shape)  # [N,B,K] 匹配 s1 的次元 K
        score_diff = sL_rep - s1_rep  # [N,B,K]

        # 聚合差值
        feat_shift = feat_diff.mean(dim=0)  # [B,F]
        score_shift = score_diff.mean(dim=0)  # [B,K] or [B,1]

        # 随机幅度
        if count is not None:
            np.random.seed(int(count))
        r = np.random.normal(loc=0.0, scale=aug_scale)

        aug_feat = feature1 + float(r) * feat_shift  # [B,F]
        aug_score = s1 + float(r) * score_shift  # [B,K] / [B,1]

        # 如果你的标签是标量 [B]，则还原回 [B]
        if score1.dim() == 1:
            aug_score = aug_score.view(-1)

        return aug_feat, aug_score
