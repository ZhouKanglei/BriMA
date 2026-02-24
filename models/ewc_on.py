# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Continual learning via" " online EWC.")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument(
        "--e_lambda", type=float, required=True, help="lambda weight for EWC"
    )
    parser.add_argument(
        "--gamma", type=float, required=True, help="gamma parameter for EWC online"
    )

    return parser


class EwcOn(ContinualModel):
    NAME = "ewc_on"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)

        self.checkpoint = None
        self.fish = None

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        self.args.gamma = 1
        # self.args.e_lambda = 1

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (
                self.fish
                * ((self.module.get_params(device=self.device) - self.checkpoint) ** 2)
            ).sum()
            return penalty

    def end_task(self, dataset):
        fish = torch.zeros_like(self.module.get_params())

        for j, batch in enumerate(dataset.train_loader):
            video = batch["video"].to(self.device)  # [B, Tv, Dv]
            audio = batch["audio"].to(self.device)  # [B, Ta, Da]
            text = batch["text"].to(self.device)  # [B, ...]
            labels = batch["labels"].to(self.device)  # [B] or [B,1]

            inputs = (video, audio, text)

            for v, a, t, lab in zip(video, audio, text, labels):
                self.opt.zero_grad()

                ex = (v.unsqueeze(0), a.unsqueeze(0), t.unsqueeze(0))

                output = self.net(ex)
                loss = -1 * self.loss(output, lab.unsqueeze(0))
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.module.get_grads() ** 2

        fish /= len(dataset.train_loader) * self.args.batch_size

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.module.get_params(device=self.device).data.clone()

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
