# -*- coding: utf-8 -*-
# @Time: 2023/6/25 20:14

import torch
from models.utils.continual_model import ContinualModel


class Adam(ContinualModel):
    NAME = "adam"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Adam, self).__init__(backbone, loss, args, transform)

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def begin_task(self, dataset):
        """
        Begin a new task with the specified dataset.

        This method can be overridden to perform any necessary operations when a new task begins.
        If no operations are needed, the method can simply pass.

        Parameters:
            dataset (Any): The dataset associated with the new task.
        """
        pass

    def end_task(self, dataset):
        """
        End the current task with the specified dataset.

        This method can be overridden to perform any necessary operations when a task ends.
        If no operations are needed, the method can simply pass.

        Parameters:
            dataset (Any): The dataset associated with the task that is ending.
        """
        pass

    def observe(
        self, inputs, labels, masks=None, not_aug_inputs=None, epoch=None, task=None
    ):
        """
        Method to perform a single optimization step on the model using the provided inputs and labels.
    
        This method computes the loss based on the model's predictions and the true labels, 
        performs backpropagation, and updates the model parameters using the Adam optimizer.
    
        Parameters:
            inputs (tuple): The input data (video, audio, text) for the model.
            labels (torch.Tensor): The target labels corresponding to the input data.
            masks (Optional[torch.Tensor]): A tensor of shape N x 3 indicating which modalities are present (default is None).
            not_aug_inputs (Optional[tuple]): The ground truth data of inputs (video, audio, text) (default is None).
            epoch (Optional[int]): An optional epoch number for tracking (default is None).
            task (Optional[int]): An optional identifier for the current task (default is None).
        """

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()

        self.opt.step()

        return loss.item()
