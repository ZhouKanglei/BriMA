#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/17 上午10:37


import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel


class SI(ContinualModel):
    NAME = "si"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(SI, self).__init__(backbone, loss, args, transform)

        self.checkpoint = self.net.get_params(device=self.device).data.clone()
        self.big_omega = None
        self.small_omega = 0
        self.xi = 1
        self.c = 1

    def penalty(self):
        self.module = self.net.module if hasattr(self.net, "module") else self.net

        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (
                self.big_omega
                * ((self.module.get_params(device=self.device) - self.checkpoint) ** 2)
            ).sum()
            return penalty

    def end_task(self, dataset):
        self.module = self.net.module if hasattr(self.net, "module") else self.net

        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(
                self.module.get_params(device=self.device)
            ).to(self.device)

        self.big_omega += self.small_omega / (
            (self.module.get_params().data - self.checkpoint) ** 2 + self.xi
        )

        # store parameters checkpoint and reset small_omega
        self.checkpoint = (
            self.module.get_params(device=self.device).data.clone().to(self.device)
        )
        self.small_omega = 0

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        self.module = self.net.module if hasattr(self.net, "module") else self.net

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.c * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.module.parameters(), 1)
        self.opt.step()

        self.small_omega += self.args.lr * self.module.get_grads().data ** 2

        return loss.item()
