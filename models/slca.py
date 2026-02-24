#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/17
import torch
import torch.nn.functional as F
from torch.distributions import Normal  # we'll use independent Normals now
from models.utils.continual_model import ContinualModel
from utils.status import ProgressBar


class SLCA(ContinualModel):
    """
    SLCA with per-modality diagonal Gaussian replay.

    We no longer store a full covariance matrix per modality because that is
    huge (e.g. 50k x 50k) and numerically singular.

    Instead, for each modality we store:
        - mean vector  [D]
        - var vector   [D]  (per-dimension variance)

    And for labels:
        - mean scalar [1]
        - var scalar  [1]

    During replay, we sample each dimension independently using Normal(mean, std),
    where std = sqrt(var + eps). Then we reshape back.
    """

    NAME = "slca"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(SLCA, self).__init__(backbone, loss, args, transform)

        # mean_cov will be a list of dicts, one per task.
        # Each dict stores per-modality stats:
        # {
        #   "video": { "mean": μ_video[Dv], "var": σ2_video[Dv] },
        #   "audio": { "mean": μ_audio[Da], "var": σ2_audio[Da] },
        #   "text":  { "mean": μ_text[Dt],  "var": σ2_text[Dt]  },
        #   "label": { "mean": μ_lbl[1],    "var": σ2_lbl[1]    }
        # }
        self.mean_cov = []

        self.current_task = 0
        self.n_tasks = args.n_tasks

        self.progress_bar = ProgressBar(verbose=True)

        self.opt = torch.optim.Adam(
            [
                {
                    "params": self.net.feature_extractor.parameters(),
                    "lr": self.args.lr * 0.1,
                },
                {"params": self.net.regressor.parameters(), "lr": self.args.lr},
            ],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        self.feature_shape = {}
        self.label_shape = None

    def _flatten_modality_batch(self, x):
        """
        x: tensor [B, ...] (e.g. [B, T, D])
        return: [B, D_flat]
        """
        return x.reshape(x.shape[0], -1)

    def compute_mean_cov(self, dataset):
        """
        For the current task, estimate per-modality diagonal Gaussians:
        mean and per-dimension variance for video/audio/text, plus label.

        We iterate over dataset.train_loader, collect all flattened features,
        then compute:
            mean = E[x]
            var  = E[(x-mean)^2]  (unbiased sample variance)
        """
        self.net.eval()

        video_list = []
        audio_list = []
        text_list = []
        label_list = []

        for _, batch in enumerate(dataset.train_loader):
            video = batch["video"].to(self.device)  # [B, Tv, Dv]
            audio = batch["audio"].to(self.device)  # [B, Ta, Da]
            text = batch["text"].to(self.device)  # [B, ...]
            labels = batch["labels"].to(self.device)  # [B] or [B,1]

            # record shapes once
            if "video" not in self.feature_shape:
                self.feature_shape["video"] = tuple(video.shape[1:])
            if "audio" not in self.feature_shape:
                self.feature_shape["audio"] = tuple(audio.shape[1:])
            if "text" not in self.feature_shape:
                self.feature_shape["text"] = tuple(text.shape[1:])
            if self.label_shape is None:
                self.label_shape = tuple(labels.shape[1:])  # usually ()

            v_flat = self._flatten_modality_batch(video)  # [B, Dv_flat]
            a_flat = self._flatten_modality_batch(audio)  # [B, Da_flat]
            t_flat = self._flatten_modality_batch(text)  # [B, Dt_flat]
            lbl = labels.reshape(labels.shape[0], -1)  # [B, 1]

            video_list.append(v_flat.detach().cpu())
            audio_list.append(a_flat.detach().cpu())
            text_list.append(t_flat.detach().cpu())
            label_list.append(lbl.detach().cpu())

        # concat: [N, D_flat]
        video_all = torch.cat(video_list, dim=0)
        audio_all = torch.cat(audio_list, dim=0)
        text_all = torch.cat(text_list, dim=0)
        label_all = torch.cat(label_list, dim=0)  # [N,1]

        def _mean_var(X):
            """
            X: [N, D]
            return mean[D], var[D]
            var is per-dimension sample variance (unbiased=True).
            """
            mean_x = torch.mean(X, dim=0)  # [D]
            var_x = torch.var(X, dim=0, unbiased=True)  # [D]
            return mean_x, var_x

        mean_video, var_video = _mean_var(video_all)
        mean_audio, var_audio = _mean_var(audio_all)
        mean_text, var_text = _mean_var(text_all)
        mean_label, var_label = _mean_var(label_all)  # D=1 here

        self.mean_cov.append(
            {
                "video": {
                    "mean": mean_video,  # [Dv_flat]
                    "var": var_video,  # [Dv_flat]
                },
                "audio": {
                    "mean": mean_audio,
                    "var": var_audio,
                },
                "text": {
                    "mean": mean_text,
                    "var": var_text,
                },
                "label": {
                    "mean": mean_label,  # [1]
                    "var": var_label,  # [1]
                },
            }
        )

    def recover_inputs_from_flat(self, video_flat, audio_flat, text_flat):
        """
        Reshape sampled flat vectors back into the original shapes.

        video_flat: [N, Dv_flat]
        -> [N] + feature_shape["video"]  e.g. [N, Tv, Dv]

        Same for audio, text.
        """
        vid_shape = self.feature_shape["video"]
        aud_shape = self.feature_shape["audio"]
        txt_shape = self.feature_shape["text"]

        N = video_flat.shape[0]

        video_rec = video_flat.reshape((N,) + vid_shape).contiguous()
        audio_rec = audio_flat.reshape((N,) + aud_shape).contiguous()
        text_rec = text_flat.reshape((N,) + txt_shape).contiguous()

        return (video_rec, audio_rec, text_rec)

    def sample(self, num_sampled_pcls=20):
        """
        Replay sampling using diagonal Gaussians.

        For each past task:
          - Sample num_sampled_pcls synthetic examples from each modality's
            independent Normal(mean, std).
          - Reshape them back to [B, T, D] etc.

        Finally concatenate all tasks' samples, shuffle, and return:
            (video_batch, audio_batch, text_batch), label_batch
        """
        all_video_flat = []
        all_audio_flat = []
        all_text_flat = []
        all_label = []

        eps = 1e-6  # numerical safety

        for task_stats in self.mean_cov:
            # get stats
            mv = task_stats["video"]["mean"]
            vv = task_stats["video"]["var"]
            ma = task_stats["audio"]["mean"]
            va = task_stats["audio"]["var"]
            mt = task_stats["text"]["mean"]
            vt = task_stats["text"]["var"]
            ml = task_stats["label"]["mean"]
            vl = task_stats["label"]["var"]

            # std = sqrt(var + eps)
            sv = torch.sqrt(vv + eps)  # [Dv_flat]
            sa = torch.sqrt(va + eps)
            st = torch.sqrt(vt + eps)
            sl = torch.sqrt(vl + eps)  # [1]

            # We sample each dimension independently:
            # Normal(mean, std).sample([num_samples, D])
            dist_v = Normal(mv, sv)  # elementwise
            dist_a = Normal(ma, sa)
            dist_t = Normal(mt, st)
            dist_l = Normal(ml, sl)

            # Sample shape:
            #   video_samp: [num_sampled_pcls, Dv_flat]
            #   audio_samp: [num_sampled_pcls, Da_flat]
            #   text_samp:  [num_sampled_pcls, Dt_flat]
            #   label_samp: [num_sampled_pcls, 1]
            video_samp = dist_v.rsample(
                (num_sampled_pcls,)
            )  # rsample -> allows grad if needed
            audio_samp = dist_a.rsample((num_sampled_pcls,))
            text_samp = dist_t.rsample((num_sampled_pcls,))
            label_samp = dist_l.rsample((num_sampled_pcls,))

            all_video_flat.append(video_samp)
            all_audio_flat.append(audio_samp)
            all_text_flat.append(text_samp)
            all_label.append(label_samp)

        # concat across tasks
        video_all = torch.cat(all_video_flat, dim=0)  # [N_total, Dv_flat]
        audio_all = torch.cat(all_audio_flat, dim=0)
        text_all = torch.cat(all_text_flat, dim=0)
        label_all = torch.cat(all_label, dim=0)  # [N_total, 1]

        # shuffle
        idx = torch.randperm(video_all.shape[0])
        video_all = video_all[idx]
        audio_all = audio_all[idx]
        text_all = text_all[idx]
        label_all = label_all[idx]

        # reshape back to original temporal shapes
        video_rec, audio_rec, text_rec = self.recover_inputs_from_flat(
            video_all, audio_all, text_all
        )

        return (video_rec, audio_rec, text_rec), label_all

    def ca(self, num_sampled_pcls=20):
        """
        Classifier / regressor alignment via replay data.
        After task > 1, we generate pseudo-samples from all previous tasks,
        then fine-tune regressor to fit those pseudo labels.
        """
        (video_batch, audio_batch, text_batch), targets = self.sample(
            num_sampled_pcls=num_sampled_pcls
        )

        # Each "block" corresponds to num_sampled_pcls samples per task.
        # We'll still loop per task so progress_bar looks similar to before.
        n_tasks_replayed = len(self.mean_cov)
        for e in range(self.args.n_epochs):
            for task_idx in range(n_tasks_replayed):
                st = task_idx * num_sampled_pcls
                ed = (task_idx + 1) * num_sampled_pcls

                v_inp = video_batch[st:ed].float().to(self.device)
                a_inp = audio_batch[st:ed].float().to(self.device)
                t_inp = text_batch[st:ed].float().to(self.device)
                lbls = targets[st:ed].float().to(self.device)  # [K,1]

                inp_tuple = (v_inp, a_inp, t_inp)
                pred = self.module.regressor(inp_tuple)

                loss = F.mse_loss(pred["output"], lbls)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.progress_bar.prog(
                    task_idx,
                    n_tasks_replayed,
                    e,
                    self.current_task - 1,
                    loss.item(),
                )
        # newline after progress bar
        print("")

    def end_task(self, dataset):
        """
        End of a task:
        - If we already have past tasks, run alignment with replay.
        - Then compute new stats for this task.
        """
        self.current_task += 1

        if self.current_task > 1:
            self.args.logging.info("Classifier alignment ...")
            self.ca(num_sampled_pcls=20)

        self.args.logging.info("Compute mean/var ...")
        self.compute_mean_cov(dataset)

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        """
        Standard supervised training step on current real batch.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()
