#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/08/30 15:38:31
import os
import json
import torch

from models.utils.continual_model import ContinualModel

from utils.status import progress_bar
from utils.metrics import stat_results
from utils.misc import save_metric_results, print_results


class Joint(ContinualModel):
    NAME = "joint"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        self.best_metric = None

    def train(self, dataset):
        # 1 initialization

        loader = dataset.joint_train_loader
        test_loader = dataset.joint_test_loader

        self.scheduler = dataset.get_scheduler(self, self.args)
        # 2 before training, evaluate the model
        self.evaluate(test_loader, epoch=0)
        # 3 train the model
        for e in range(self.args.n_epochs):
            # initialize the total loss
            total_loss = -1
            label_scores, pred_scores = [], []
            # batch training
            for i, batch in enumerate(loader):
                # forward
                inputs = (
                    batch['video'].to(self.args.output_device),
                    batch['audio'].to(self.args.output_device),
                    batch['text'].to(self.args.output_device),
                )
                labels = batch['labels'].to(self.args.output_device)

                outputs = self.net(inputs)

                self.opt.zero_grad()

                loss = self.loss(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                self.opt.step()

                if self.args.model == "base":
                    progress_bar(i, len(loader), e, "B", loss.item())
                else:
                    progress_bar(i, len(loader), e, "J", loss.item())

                label_scores.extend(labels.detach().cpu().numpy().flatten().tolist())
                pred_scores.extend(
                    outputs["output"].detach().cpu().numpy().flatten().tolist()
                )

            # denormalize the scores
            label_scores = [
                i * self.args.dataset_args["score_range"] for i in label_scores
            ]
            pred_scores = [
                i * self.args.dataset_args["score_range"] for i in pred_scores
            ]

            # learning rate scheduling
            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()
                print()
                self.args.logging.info(
                    "The current learning rate is {:.06f}".format(lr[0])
                )
                self.scheduler.step()
            else:
                print()

            # print the average loss of the epoch
            avg_loss = total_loss / len(loader)
            self.args.logging.info(f"Average training loss is {avg_loss:.4f}")
            # stat the performance on the training set
            train_results = stat_results(
                pred_scores,
                label_scores,
                f"Train - Epoch {e + 1}",
                self.args.logging,
            )
            # evaluate the model on the test set
            results = self.evaluate(test_loader, epoch=e)
            # save best model
            self.save_best_model(results, epoch=e)

        # 4 print the best model information
        print_results(
            self.best_metric,
            f"\033[93m Best epoch {self.best_metric['epoch']}\033[0m",
            logger=self.args.logging,
        )
        # 5 save results to json file
        save_metric_results(
            self.best_metric,
            self.args.output_result_dir,
            filename="best_results.json",
            logger=self.args.logging,
        )

    def evaluate(self, test_loader, epoch=None):
        # set the model to evaluation mode
        status = self.net.training
        self.net.eval()
        # initialize the label and output scores
        label_scores, pred_scores = [], []
        # batch evaluation
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                inputs = (
                    batch['video'].to(self.args.output_device),
                    batch['audio'].to(self.args.output_device),
                    batch['text'].to(self.args.output_device),
                )
                labels = batch['labels'].to(self.args.output_device)

                outputs = self.net(inputs)

                label_scores.extend(labels.detach().cpu().numpy().flatten().tolist())
                pred_scores.extend(
                    outputs["output"].detach().cpu().numpy().flatten().tolist()
                )

        # denormalize the scores
        label_scores = [i * self.args.dataset_args["score_range"] for i in label_scores]
        pred_scores = [i * self.args.dataset_args["score_range"] for i in pred_scores]

        # compute the metrics
        results = stat_results(
            pred_scores, label_scores, f" Test - Epoch {epoch + 1}", self.args.logging
        )

        # set the model back to the original status
        self.net.train(status)

        return results

    def save_best_model(self, metrics, epoch=None):
        if self.best_metric is None:
            self.best_metric = metrics
            self.best_metric["epoch"] = epoch + 1

        if metrics["SRCC"] >= self.best_metric["SRCC"]:
            self.best_metric = metrics
            self.best_metric["epoch"] = epoch + 1
            self.save_model("best_model.pth")
            # log the saving information
            self.args.logging.info(
                f"Best model saved successfully".center(36, " ").center(80, "-")
            )
            self.args.logging.info(
                "Current best SRCC is \033[93m{:.4f}\033[0m ({:d}-th epoch)".format(
                    self.best_metric["SRCC"], self.best_metric["epoch"]
                )
            )
            self.args.logging.info("".center(80, "-"))
        else:
            metrics.update({"epoch": epoch + 1})
            self.save_model("last_model.pth", metrics)

    def save_model(self, model_name, metrics=None):
        weight_path = os.path.join(self.args.output_model_weight_dir, model_name)
        # save the model
        state_dict = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
        }
        state_dict.update(self.best_metric if metrics is None else metrics)

        torch.save(state_dict, weight_path)
