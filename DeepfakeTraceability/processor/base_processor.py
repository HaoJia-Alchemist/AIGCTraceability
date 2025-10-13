from abc import ABC, abstractmethod
import logging
import os.path as osp

import torch
from accelerate import Accelerator

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


class BaseProcessor(ABC):
    """
    """

    def __init__(
            self,
            config,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_fn,
            num_query,
            writer=None
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or scheduler is None or writer is None or loss_fn is None or num_query is None or center_criterion is None or train_loader is None or val_loader is None:
            raise NotImplementedError(
                "config, model, optimizer, optimizer_center, scheduler, loss_fn, num_query, center_criterion, train_loader, val_loader, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.center_criterion = center_criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_query = num_query
        self.writer = writer
        self.accelerator = Accelerator()

    def do_train(self):
        """
        Train the model.
        """
        pass

    def train_epoch(self,  epoch):
        """
        Train the model for one epoch.
        """
        pass

    def train_step(self, data):
        """
        Train the model for one step.
        """
        pass

    def do_validate(self):
        """
        Validate the model.
        """
        pass

    def do_inference(self):
        """
        Inference the model.
        """
        pass

    def inference(self):
        """
        Inference the model.
        """
        pass

    def save_model(self):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrap_model.cpu().state_dict(), osp.join(self.config["logging"]["log_dir"], "best_model", "model.pth"))
        self.logger.info("Saved model to {}".format(osp.join(self.config["logging"]["log_dir"], "best_model", "model.pth")))

    def load_model(self, model_path):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        unwrap_model.load_state_dict(torch.load(model_path))
        self.model = self.accelerator.prepare(unwrap_model)

    def save_state(self):
        self.accelerator.save_state(osp.join(self.config["logging"]["log_dir"], "checkpoint"), total_limit=self.config["logging"]["checkpoints_total_limit"])
        self.logger.info("Saved state to {}".format(osp.join(self.config["logging"]["log_dir"], "checkpoint")))

    def load_state(self, checkpoint_dir):
        self.accelerator.load_state(checkpoint_dir)