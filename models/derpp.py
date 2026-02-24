#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/7/5 下午11:44

import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer


class Derpp(ContinualModel):
    NAME = "derpp"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(
            self.args.buffer_size,
            "cpu",
            store_logits=True,
        )

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        if not self.buffer.is_empty() and task:
            # get data from buffer
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )

            buf_inputs = tuple(input.to(self.device) for input in buf_inputs)
            buf_logits = buf_logits.to(self.device)

            self.opt.zero_grad()
            buf_outputs = self.net(buf_inputs)
            loss = self.args.alpha * F.mse_loss(buf_outputs["output"], buf_logits)
            loss.backward()
            self.opt.step()
            # get data from buffer
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )

            buf_inputs = tuple(input.to(self.device) for input in buf_inputs)
            buf_labels = buf_labels.to(self.device)

            self.opt.zero_grad()
            buf_outputs = self.net(buf_inputs)
            loss = self.args.beta * F.mse_loss(buf_outputs["output"], buf_labels)

            loss.backward()
            self.opt.step()

        if not epoch:  # epoch == 0
            not_aug_inputs = tuple(input.cpu() for input in not_aug_inputs)
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels.cpu(),
                logits=outputs["output"].data.cpu(),
            )

        return loss.item()
