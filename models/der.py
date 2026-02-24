#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/7/2 下午3:42
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer


class Der(ContinualModel):
    NAME = "der"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, "cpu", store_logits=True)

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

        if not self.buffer.is_empty() and task:  # task > 0
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )

            buf_inputs = tuple(input.to(self.device) for input in buf_inputs)

            buf_outputs = self.net(buf_inputs)
            loss = loss + self.args.alpha * F.mse_loss(
                buf_outputs['output'], buf_logits.to(self.device)
            )

        loss.backward()
        self.opt.step()

        if not epoch:  # epoch == 0
            not_aug_inputs = tuple(input.cpu() for input in not_aug_inputs)
            self.buffer.add_data(
                examples=not_aug_inputs, logits=outputs["output"].data.cpu()
            )

        return loss.item()
