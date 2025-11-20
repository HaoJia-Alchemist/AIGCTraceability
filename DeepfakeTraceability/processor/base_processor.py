import os
import os.path as osp
from abc import ABC

import torch
from accelerate import Accelerator


class BaseProcessor(ABC):
    """
    """

    def __init__(
            self,
            config,
            model,
            center_criterion=None, train_loader=None, val_loader=None, optimizer=None, optimizer_center=None,
            scheduler=None, loss_fn=None, num_query=None, accelerator=None
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or accelerator is None:
            raise NotImplementedError(
                "config, model, accelerator")

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
        self.accelerator = accelerator

    def do_train(self):
        """
        Train the model.
        """
        pass

    def train_epoch(self, epoch):
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

    def save_model(self):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        if not osp.exists(osp.join(self.config["logging"]["log_dir"], "best_model")):
            os.makedirs(osp.join(self.config["logging"]["log_dir"], "best_model"))
        torch.save(unwrap_model.state_dict(), osp.join(self.config["logging"]["log_dir"], "best_model", "model.pth"))
        self.logger.info(
            "Saved model to {}".format(osp.join(self.config["logging"]["log_dir"], "best_model", "model.pth")))

    def load_model(self, model_path):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        unwrap_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.accelerator.prepare(unwrap_model)

    def save_state(self):
        self.accelerator.save_state(osp.join(self.config["logging"]["log_dir"], "checkpoint"),
                                    total_limit=2)
        self.logger.info("Saved state to {}".format(osp.join(self.config["logging"]["log_dir"], "checkpoint")))

    def load_state(self, checkpoint_dir):
        self.accelerator.load_state(checkpoint_dir)
