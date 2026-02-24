#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/7/9 上午10:36


import torch

from models.utils.continual_model import ContinualModel
from utils.mmbuffer import MultiModalBuffer as Buffer
from utils.misc import distributed_concat


class Mer(ContinualModel):
    NAME = "mer"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Mer, self).__init__(backbone, loss, args, transform)
        # import pdb; pdb.set_trace()
        self.buffer = Buffer(self.args.buffer_size, "cpu")
        # assert args.batch_size == 1, 'Mer only works with batch_size=1'
        if hasattr(self.args, "batch_num"):
            self.batch_num = self.args.batch_num
        else:
            self.batch_num = 3
        self.batch_num = 1

    def draw_batches(self, inp, lab):
        batches = []
        for i in range(self.batch_num):
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )

                buf_inputs = tuple(b.to(self.device) for b in buf_inputs)
                inp = tuple(i.to(self.device) for i in inp)

                inputs = tuple(torch.cat((buf_inp, inp_j)) for buf_inp, inp_j in zip(buf_inputs, inp))
                labels = torch.cat((buf_labels.to(self.device), lab.to(self.device)))
                batches.append((inputs, labels))
            else:

                inp = tuple(i.to(self.device) for i in inp)

                batches.append((inp, lab))
        return batches

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        batches = self.draw_batches(inputs, labels)
        theta_A0 = self.module.get_params(device=self.device).data.clone()

        for i in range(self.batch_num):
            theta_Wi0 = self.module.get_params(device=self.device).data.clone()

            batch_inputs, batch_labels = batches[i]

            # within-batch step
            self.opt.zero_grad()
            outputs = self.net(batch_inputs)
            loss = self.loss(outputs, batch_labels)
            loss.backward()
            self.opt.step()

            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (
                self.module.get_params(device=self.device) - theta_Wi0
            )
            self.module.set_params(new_params)

        if not epoch:  # epoch == 0
            not_aug_inputs = tuple(i.to(self.device) for i in not_aug_inputs)
            self.buffer.add_data(examples=not_aug_inputs, labels=labels.cpu())

        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (
            self.module.get_params(device=self.device) - theta_A0
        )
        self.module.set_params(new_new_params)

        return loss.item()
