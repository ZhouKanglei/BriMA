#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/17 上午11:11


import torch
from datasets import get_dataset
from torch.optim import SGD, Adam
import torch.nn.functional as F
import copy


from models.utils.continual_model import ContinualModel
from utils.args import (
    add_management_args,
    add_experiment_args,
    add_rehearsal_args,
    ArgumentParser,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Continual learning via" " Learning without Forgetting."
    )
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--alpha", type=float, default=0.5, help="Penalty weight.")
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=2,
        help="Temperature of the softmax function.",
    )
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = "lwf"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.current_task = 0

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = Adam(self.module.regressor.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, batch in enumerate(dataset.train_loader):
                    video = batch["video"].to(self.device)  # [B, Tv, Dv]
                    audio = batch["audio"].to(self.device)  # [B, Ta, Da]
                    text = batch["text"].to(self.device)  # [B, ...]

                    inputs = (video, audio, text)
                    labels = batch["labels"].to(self.device)  # [B] or [B,1]e)

                    opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels)
                    loss.backward()
                    opt.step()

            # clone the model
            self.old_net = copy.deepcopy(self.net)

        self.current_task += 1

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if self.old_net is not None:
            self.old_net.eval()
            with torch.no_grad():
                logits = self.old_net(inputs)
            loss += self.args.alpha * F.mse_loss(
                outputs["output"], logits["output"].data
            )

        loss.backward()
        self.opt.step()

        return loss.item()
