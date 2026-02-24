#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/4 上午10:02

import copy
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer
from utils.metrics import ListNetLoss


# ---------------------- helpers ----------------------


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


def _to_device_inputs(inp_tuple, device):
    return tuple(x.to(device) for x in inp_tuple)


def _slice_inputs(inp_tuple, idx):
    return tuple(x[idx] for x in inp_tuple)


def _pack_features(feats):
    """
    把多模态特征打包成 [B, F]；同时记录每个模态的原始 shape 以便恢复。
    支持：
      - tuple/list: (f_v, f_a, f_t)
      - dict: {"video":..., "audio":..., "text":...}
      - tensor: 单模态
    返回:
      flat: [B, F]
      meta: [{"name": <str>, "shape": tuple}, ...]   # 记录每个模态原始 shape（含 batch 维）
      order: list[str] 模态顺序（用于解包为 tuple 时的顺序）
    """
    parts = []
    meta = []
    order = []

    def _append(name, t):
        parts.append(t.reshape(t.shape[0], -1))
        meta.append({"name": name, "shape": tuple(t.shape)})
        order.append(name)

    if isinstance(feats, (tuple, list)):
        # 假定顺序就是 (video, audio, text)
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
        raise TypeError(f"Unsupported features type for packing: {type(feats)}")

    if len(parts) == 0:
        raise ValueError("No valid modality features to pack.")

    flat = torch.cat(parts, dim=1)
    return flat, meta, order


def _unpack_features(flat, meta, order):
    """
    将 [B, F] 的扁平特征根据 meta 和 order 还原为 (visual, audio, text) 的 tuple。
    注意：要求 projector 输出维度等于 pack 后的总维度，否则无法还原。
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

    # 返回 tuple，顺序与 order 保持一致；对 regressor 需要 (visual, audio, text)
    out_tuple = tuple(outs_by_name[nm] for nm in order if nm in outs_by_name)
    return out_tuple


# ---------------------- MAGR (multimodal buffer) ----------------------


class Magr(ContinualModel):
    NAME = "fea_gr_mm"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Magr, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, device="cpu")
        self.buffer.empty()

        self.current_task = 0
        self.lambda1 = args.alpha  # projector 对齐权重
        self.lambda2 = args.beta  # 图正则权重
        self.n_tasks = (
            args.n_tasks + 1 if getattr(args, "fewshot", False) else args.n_tasks
        )

        # 仅优化 projector + regressor（backbone 固定）
        self.opt = torch.optim.Adam(
            params=[
                {"params": self.net.projector.parameters(), "lr": self.args.lr},
                {"params": self.net.regressor.parameters(), "lr": self.args.lr},
            ],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        self.graph_reg_loss = ListNetLoss()
        self.old_feature_extractor = None  # 仅前向用，辅助 projector 对齐

    # ------------------ 选样写入多模态缓冲（写 inputs+labels） ------------------

    @staticmethod
    def _select_by_rank_even(labels_1d: torch.Tensor, k: int):
        n = labels_1d.numel()
        if k >= n:
            return list(range(n))
        order = torch.argsort(labels_1d)
        picks = _evenly_spaced_indices(n, k)
        return order[picks].tolist()

    def inputs2buffer_select(self, dataset):
        examples_per_task = max(1, self.args.buffer_size // max(1, (self.n_tasks - 1)))
        self.args.logging.info(
            f"Current task {self.current_task} - select {examples_per_task} samples"
        )

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
                sel_inputs = _slice_inputs(inputs, idx)
                sel_labels = labels[idx].detach().cpu()

                self.buffer.add_data(
                    examples=tuple(x.detach().cpu() for x in sel_inputs),
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

    # ------------------ 任务收尾 ------------------

    def end_task(self, dataset):
        self.current_task += 1
        if self.current_task < self.n_tasks:
            self.inputs2buffer_select(dataset)

        # 备份当前特征抽取器（仅前向）
        try:
            self.old_feature_extractor = copy.deepcopy(
                self.net.feature_extractor
            ).eval()
            for p in self.old_feature_extractor.parameters():
                p.requires_grad = False
        except Exception:
            self.old_feature_extractor = None

    # ------------------ 训练步 ------------------

    def observe(self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None):
        """
        稳健版：
        - 对 features / labels 做 nan_to_num
        - 各分项 loss 检查 isfinite；不合法则跳过该项并记录日志
        - 图正则前对特征做标准化，避免极端值
        - 反向前做梯度裁剪，避免爆炸
        """
        def _finite(x):  # 张量是否全是有限数
            return torch.isfinite(x).all()

        def _nanfix(x):  # 将 NaN/Inf 修正为有限数
            return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        def _safe_add(total, term, name):
            if term is None:
                return total
            if not torch.is_tensor(term):
                self.args.logging.warning(f"[{name}] is not a tensor, skip.")
                return total
            if not _finite(term):
                self.args.logging.warning(f"[{name}] has NaN/Inf, skip this term.")
                return total
            return total + term

        self.opt.zero_grad()

        # 前向
        outputs, features = self.net(inputs, returnt="all")
        # 清理 labels
        labels = _nanfix(labels).to(self.device)

        # 打包当前特征为 flat，修正数值
        try:
            cur_flat, cur_meta, cur_order = _pack_features(features)
            cur_flat = _nanfix(cur_flat).to(self.device)
        except Exception as e:
            self.args.logging.error(f"pack current features failed: {e}")
            return torch.tensor(0.0, device=self.device).item()

        # 主任务损失（你的 self.loss 会从 outputs['output'] 取预测）
        loss = torch.zeros((), device=self.device)
        try:
            loss_d = self.loss(outputs, labels)
            loss_d = _nanfix(loss_d)
            loss = _safe_add(loss, loss_d, "task_loss")
        except Exception as e:
            self.args.logging.error(f"task loss failed: {e}")

        # projector 对齐
        if self.old_feature_extractor is not None:
            with torch.no_grad():
                self.old_feature_extractor.eval()
                old_features = self.old_feature_extractor(inputs)
            try:
                old_flat, _, _ = _pack_features(old_features)
                old_flat = _nanfix(old_flat).to(self.device)
                feat_hat = self.net.projector(old_flat)
                feat_hat = _nanfix(feat_hat)

                if feat_hat.shape[1] != cur_flat.shape[1]:
                    self.args.logging.warning(
                        f"[proj_align] dim mismatch: {feat_hat.shape[1]} vs {cur_flat.shape[1]}, skip align."
                    )
                else:
                    loss_align = F.mse_loss(feat_hat, cur_flat.detach())
                    loss_align = _nanfix(loss_align)
                    loss = _safe_add(loss, self.lambda1 * loss_align, "proj_align")
            except Exception as e:
                self.args.logging.warning(f"proj align failed: {e}")

        # 回放
        if not self.buffer.is_empty():
            try:
                ((v_b, a_b, t_b), buf_labels, *_) = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_inputs = (v_b.to(self.device), a_b.to(self.device), t_b.to(self.device))
                buf_labels = _nanfix(buf_labels).to(self.device)

                buf_outputs, buf_features = self.net(buf_inputs, returnt="all")
                buf_flat, buf_meta, buf_order = _pack_features(buf_features)
                buf_flat = _nanfix(buf_flat).to(self.device)

                # 投影到当前空间
                buf_flat_hat = self.net.projector(buf_flat)
                buf_flat_hat = _nanfix(buf_flat_hat)

                if buf_flat_hat.shape[1] != cur_flat.shape[1]:
                    self.args.logging.warning(
                        f"[graph/replay] projector out dim {buf_flat_hat.shape[1]} != cur dim {cur_flat.shape[1]}, skip reg+replay."
                    )
                else:
                    # -------- Graph 正则（先标准化，避免极端数值）--------
                    eps = 1e-6
                    jf = torch.cat([buf_flat_hat, cur_flat], dim=0)
                    jl = torch.cat([buf_labels, labels], dim=0)

                    # 标准化到零均值单位方差
                    mean = jf.mean(dim=0, keepdim=True)
                    std = jf.std(dim=0, keepdim=True) + eps
                    jf_norm = _nanfix((jf - mean) / std)

                    try:
                        loss_graph = self.graph_reg_loss(
                            jf_norm, jl, blocking=self.args.minibatch_size
                        )
                        loss_graph = _nanfix(loss_graph)
                        loss = _safe_add(loss, self.lambda2 * loss_graph, "graph_reg")
                    except Exception as e:
                        self.args.logging.warning(f"graph reg failed: {e}")

                    # -------- 回放监督：需要把 flat_hat 还原为三模态喂 regressor --------
                    try:
                        buf_tuple_hat = _unpack_features(buf_flat_hat, buf_meta, buf_order)
                        buf_pred = self.net.regressor(buf_tuple_hat)
                        if isinstance(buf_pred, dict) and "output" in buf_pred:
                            buf_pred = buf_pred["output"]
                        buf_pred = _nanfix(buf_pred)

                        # 形状对齐
                        tgt = buf_labels.float().to(buf_pred.device)
                        if buf_pred.dim() == 2 and buf_pred.size(1) == 1 and tgt.dim() == 1:
                            tgt = tgt.unsqueeze(1)
                        elif buf_pred.dim() == 1 and tgt.dim() == 2 and tgt.size(1) == 1:
                            buf_pred = buf_pred.unsqueeze(1)

                        loss_replay = F.mse_loss(buf_pred, tgt)
                        loss_replay = _nanfix(loss_replay)
                        loss = _safe_add(loss, loss_replay, "replay_sup")
                    except Exception as e:
                        self.args.logging.warning(f"replay supervision failed: {e}")

            except Exception as e:
                self.args.logging.warning(f"buffer forward failed: {e}")

        # 若总损失仍然不合法，跳过该 batch 以免训练中断
        if not torch.is_tensor(loss) or not torch.isfinite(loss).all():
            self.args.logging.error(
                "Total loss is NaN/Inf — skip this batch. "
                "Tip: enable anomaly detection or lower lr; check labels/features for NaNs."
            )
            # 可选：开启一次性异常检测，定位源头（开销较大）
            # torch.autograd.set_detect_anomaly(True)
            return 0.0

        # 反向 + 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.opt.step()

        # （可选）在线写入未增强样本
        if epoch == 0 and not_aug_inputs is not None:
            self.buffer.add_data(
                examples=tuple(x.detach().cpu() for x in not_aug_inputs),
                labels=labels.detach().cpu()
            )

        return loss.item()
