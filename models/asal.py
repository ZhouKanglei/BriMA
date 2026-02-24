#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/09/06 15:33:57
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer


# ----------------------------- helpers -----------------------------


def _to_device_inputs(inp_tuple, device):
    """Move a (video, audio, text) tuple to device."""
    return tuple(x.to(device) for x in inp_tuple)


def _slice_inputs(inp_tuple, idx):
    """Slice a (video, audio, text) tuple by indices tensor/list."""
    return tuple(x[idx] for x in inp_tuple)


def _iter_modalities(features):
    """
    Yield modality tensors from features in a unified way.
    features can be:
      - tuple/list: (f_video, f_audio, f_text)
      - dict: {'video': ..., 'audio': ..., 'text': ...}
      - tensor: single modality
    """
    if isinstance(features, (tuple, list)):
        yield from features
    elif isinstance(features, dict):
        for k in ("video", "audio", "text"):
            if k in features and features[k] is not None:
                yield features[k]
    elif torch.is_tensor(features):
        yield features
    else:
        return


def _repack_modalities_like(features, tensors_list):
    """
    Pack tensors_list back to the same structure type as 'features'.
    Used only if the structure needs to be aligned (this implementation is only consumed internally and does not return).
    """
    if isinstance(features, (tuple, list)):
        return type(features)(tensors_list)
    if isinstance(features, dict):
        out = {}
        i = 0
        for k in ("video", "audio", "text"):
            if k in features and features[k] is not None:
                out[k] = tensors_list[i]
                i += 1
        return out
    if torch.is_tensor(features):
        return tensors_list[0]
    return tensors_list


# ----------------------------- compressors -----------------------------


class TemporalCompressor(nn.Module):
    """
    Compress & decompress along temporal dimension for inputs shaped [B, T, D].
    encode: [B, T, D] -> [B, K, D]   via Linear(T -> K) applied on last dim after permute
    decode: [B, K, D] -> [B, T, D]   via Linear(K -> T)
    """

    def __init__(self, T: int, K: int):
        super().__init__()
        self.enc = nn.Linear(T, K, bias=False)
        self.dec = nn.Linear(K, T, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x_t = x.permute(0, 2, 1)  # [B, D, T]
        z = self.enc(x_t)  # [B, D, K]
        z_t = z.permute(0, 2, 1)  # [B, K, D]
        # reconstruct
        x_rec_t = self.dec(z)  # [B, D, T]
        x_rec = x_rec_t.permute(0, 2, 1)  # [B, T, D]
        return z_t, x_rec


class VectorCompressor(nn.Module):
    """
    Compress & decompress along feature dimension for vectors shaped [B, D].
    encode: [B, D] -> [B, Kd]
    decode: [B, Kd] -> [B, D]
    """

    def __init__(self, D: int, Kd: int):
        super().__init__()
        self.enc = nn.Linear(D, Kd, bias=False)
        self.dec = nn.Linear(Kd, D, bias=False)

    def forward(self, x):
        # x: [B, D]
        z = self.enc(x)  # [B, Kd]
        x_rec = self.dec(z)  # [B, D]
        return z, x_rec


# ----------------------------- ASAL -----------------------------


class Asal(ContinualModel):
    NAME = "asal"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Asal, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, "cpu")

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # compression settings
        self.K_time = getattr(
            args, "num_key_frames", 8
        )  # target time steps for temporal comp.
        self.K_text = getattr(
            args, "text_latent_dim", None
        )  # if None, lazy set to min(D, K_time)

        # lazy modality compressors for FEATURE tensors (not inputs)
        self._feat_temporal_compressors = []  # list aligned with _iter_modalities order
        self._feat_vector_compressors = []

    # ---------------------- sample selection by score ----------------------

    @staticmethod
    def select_samples_by_scores(all_names, all_scores, num_per_task):
        """
        Evenly pick by rank on scores (ascending).
        """
        scores = np.array([float(s) for s in all_scores])
        names = np.array(list(all_names))
        order = np.argsort(scores)
        if len(scores) <= num_per_task:
            sel = order
        else:
            idx = np.linspace(0, len(scores) - 1, num_per_task).round().astype(int)
            sel = order[idx]
        return set(names[sel].tolist())

    # ---------------------- push selected raw samples into buffer ----------------------

    def add_samples_to_buffer(self, dataset):
        """
        Select representative samples by label score and push RAW multimodal inputs to buffer.
        No key-frames; no external feature extraction.
        """
        num_per_task = max(
            1, self.args.buffer_size // max(1, getattr(dataset, "N_TASKS", 1))
        )

        train_loader = dataset.train_loader
        all_scores, all_names = [], []
        # 1) collect names + scores
        for batch in train_loader:
            if isinstance(batch, dict):
                labels = batch["labels"]
                names = batch.get("names", [str(i) for i in range(labels.shape[0])])
            else:
                _, labels, names = batch
            all_names.extend(list(names))
            all_scores.extend(labels.reshape(-1).cpu().tolist())

        # 2) select names
        selected_names = self.select_samples_by_scores(
            all_names, all_scores, num_per_task
        )

        # 3) write selected raw inputs into buffer
        selected_cnt = 0
        for batch in train_loader:
            if isinstance(batch, dict):
                inputs = (batch["video"], batch["audio"], batch["text"])
                labels = batch["labels"]
                names = batch.get("names", [str(i) for i in range(labels.shape[0])])
            else:
                inputs, labels, names = batch

            names = list(names)
            pick_idx = [i for i, nm in enumerate(names) if nm in selected_names]
            if len(pick_idx) == 0:
                continue

            idx_t = torch.tensor(pick_idx, dtype=torch.long)
            sel_inputs = _slice_inputs(inputs, idx_t)
            sel_labels = labels[idx_t].detach().cpu()

            self.buffer.add_data(
                examples=tuple(x.detach().cpu() for x in sel_inputs),
                labels=sel_labels,
                task_labels=torch.full(
                    (sel_labels.shape[0],),
                    getattr(dataset, "i", 0) + 1,
                    dtype=torch.long,
                ),
            )
            selected_cnt += len(pick_idx)

            # remove consumed names to avoid duplicate picks
            for nm in [names[i] for i in pick_idx]:
                if nm in selected_names:
                    selected_names.remove(nm)
            if len(selected_names) == 0:
                break

        self.args.logging.info(f"Add {selected_cnt} samples to the buffer.")

    def end_task(self, dataset):
        self.add_samples_to_buffer(dataset)

        # statistics
        ret = self.buffer.get_all_data()
        # ((v,a,t), labels, [logits]?, [task_labels]?, [masks]?)
        task_labels = None
        if len(ret) >= 3 and ret[2] is not None and ret[2].dtype == torch.long:
            task_labels = ret[2]
        elif len(ret) >= 4 and ret[3] is not None and ret[3].dtype == torch.long:
            task_labels = ret[3]
        if task_labels is not None:
            for ttl in task_labels.unique():
                if int(ttl) == 0:  # skip placeholder
                    continue
                cnt = int((task_labels == ttl).sum())
                self.args.logging.info(
                    f"Task {int(ttl)} has {cnt} samples in the buffer."
                )

    # ---------------------- feature compression regularization ----------------------

    def _ensure_feat_compressors(self, features):
        """
        Lazy-create compressors matched to current FEATURES structure & shapes.
        Keeps order consistent with _iter_modalities(features).
        """
        if (
            len(self._feat_temporal_compressors) > 0
            or len(self._feat_vector_compressors) > 0
        ):
            return  # already initialized

        temp_list, vec_list = [], []
        for f in _iter_modalities(features):
            if f.dim() == 3:
                # [B, T, D] -> temporal compressor with K_time
                T = f.shape[1]
                K = min(self.K_time, T)
                temp_list.append(TemporalCompressor(T, K).to(self.device))
                vec_list.append(None)
            elif f.dim() == 2:
                # [B, D] -> vector compressor with Kd
                D = f.shape[1]
                Kd = min(D, self.K_time) if self.K_text is None else min(D, self.K_text)
                temp_list.append(None)
                vec_list.append(VectorCompressor(D, Kd).to(self.device))
            else:
                # not supported; skip
                temp_list.append(None)
                vec_list.append(None)

        self._feat_temporal_compressors = nn.ModuleList(
            [m for m in temp_list if m is not None]
        )
        self._feat_vector_compressors = nn.ModuleList(
            [m for m in vec_list if m is not None]
        )
        # To save a parallel table of "positional information" for sequential indexing
        self._feat_kind = []  # 'temporal' or 'vector' or 'none'
        ti = vi = 0
        for f in _iter_modalities(features):
            if f.dim() == 3 and ti < len(self._feat_temporal_compressors):
                self._feat_kind.append(("temporal", ti))
                ti += 1
            elif f.dim() == 2 and vi < len(self._feat_vector_compressors):
                self._feat_kind.append(("vector", vi))
                vi += 1
            else:
                self._feat_kind.append(("none", -1))

    def _feat_recon_loss(self, features):
        """
        Apply encode-decode per modality feature and compute reconstruction loss sum.
        """
        self._ensure_feat_compressors(features)

        losses = []
        ti = vi = 0
        for f, (kind, idx) in zip(_iter_modalities(features), self._feat_kind):
            if kind == "temporal":
                comp = self._feat_temporal_compressors[idx]
                _, f_rec = comp(f)
                losses.append(F.mse_loss(f_rec, f))
            elif kind == "vector":
                comp = self._feat_vector_compressors[idx]
                _, f_rec = comp(f)
                losses.append(F.mse_loss(f_rec, f))
            else:
                continue

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    # ---------------------- training ----------------------

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        self.opt.zero_grad()

        # forward
        outputs, features = self.net(
            inputs, returnt="all"
        )  # outputs: dict with 'output'
        # Main task loss (your loss will use outputs['output'] internally)
        loss = self.loss(outputs, labels)

        # Feature compression-decompression reconstruction regularization
        feat_rec_loss = self._feat_recon_loss(features)
        loss = loss + self.args.beta * feat_rec_loss

        loss.backward()
        self.opt.step()

        # Replay
        if (not self.buffer.is_empty()) and task:
            ((v_b, a_b, t_b), buf_labels, *_) = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_inputs = (v_b.to(self.device), a_b.to(self.device), t_b.to(self.device))
            buf_labels = buf_labels.to(self.device)

            self.opt.zero_grad()
            buf_outputs, _ = self.net(buf_inputs, returnt="all")
            # Supervised replay (prediction vs label)
            replay_loss = F.mse_loss(
                buf_outputs["output"].float(),
                buf_labels.float().view_as(buf_outputs["output"]).to(self.device),
            )
            replay_loss.backward()
            self.opt.step()

        # Write the current batch's non-augmented samples into the buffer (can be adjusted as needed)
        if epoch == 0:
            self.buffer.add_data(
                examples=tuple(x.detach().cpu() for x in not_aug_inputs),
                labels=labels.detach().cpu(),
            )

        return loss.item()
