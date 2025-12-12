import logging
import time
from datetime import timedelta

import torch
from tqdm import tqdm

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from . import BaseProcessor
from . import PROCESSOR_FACTORY

logger = logging.getLogger(__name__)

@PROCESSOR_FACTORY.register_module("base")
class Processor(BaseProcessor):
    def __init__(self, config, model, train_loader=None, val_loader=None, optimizer=None,
                 scheduler=None, loss_fn=None, num_query=None, accelerator=None):
        super(Processor, self).__init__(config, model, train_loader, val_loader, optimizer,
                                        scheduler, loss_fn, num_query, accelerator)
        self.log_period = self.config["train"]["log_period"]
        self.checkpoint_period = self.config["train"]["checkpoint_period"]
        self.eval_period = self.config["train"]["eval_period"]
        self.max_epoch =getattr(self.config["train"], "max_epoch", None)
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.epoch = self.model.epoch if self.accelerator.num_processes == 1 else self.model.module.epoch
        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=config["test"]["feat_norm"], reranking=self.config["test"]["re_ranking"])
        logger.info('processor init ...')
        logger.info(
            f'log_period: {self.log_period}, checkpoint_period: {self.checkpoint_period}, eval_period: {self.eval_period}, max_epoch: {self.max_epoch}')

    def do_train(self):
        logger.info('start training')
        all_start_time = time.monotonic()
        logger.info("model: {}".format(self.model))
        best_metrics = {"mAP": 0.0, "Rank@1": 0.0, "Rank@5": 0.0, "Rank@10": 0.0, "Epoch": 0}
        for epoch in range(self.epoch, self.max_epoch):
            self.train_epoch(epoch)
            if epoch % self.checkpoint_period == 0:
                if self.accelerator.is_main_process:
                    self.save_state()
            if epoch % self.eval_period == 0 or epoch == self.max_epoch - 1:
                cmc, mAP = self.do_validate()
                if mAP > best_metrics["mAP"]:
                    best_metrics = {"mAP": mAP, "Rank@1": cmc[0], "Rank@5": cmc[4], "Rank@10": cmc[9], "Epoch": epoch}
                    logger.info(
                        "===> Best mAP: {:.3f}, Rank@1: {:.3f}, Rank@5: {:.3f}, Rank@10: {:.3f}, Epoch: {}".format(
                            best_metrics["mAP"], best_metrics["Rank@1"], best_metrics["Rank@5"],
                            best_metrics["Rank@10"], best_metrics["Epoch"]))
                    if self.accelerator.is_main_process:
                        self.save_model()
            self.accelerator.wait_for_everyone()
            self.epoch += 1
        all_end_time = time.monotonic()
        total_time = timedelta(seconds=all_end_time - all_start_time)
        logger.info("Total running time: {}".format(total_time))
        return best_metrics

    def train_epoch(self, epoch):
        logger.info("===> Epoch[{}] start!".format(epoch))
        start_time = time.time()
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.model.train()
        for n_iter, data in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
            self.train_step(data)
            if (n_iter + 1) % self.log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                 .format(epoch, (n_iter + 1), len(self.train_loader),
                                         self.loss_meter.avg, self.acc_meter.avg, self.scheduler.get_lr()[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / len(self.train_loader)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                         .format(epoch, time_per_batch, self.config["dataset"]["train_batch_size"] / time_per_batch))

    def train_step(self, data_dict):
        # logger.info(f"Process_ID:{self.accelerator.process_index}, df_id:{df_id}, img_path: {list(zip(df_name,img_path))}")
        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            output_dict = self.model(data_dict)
            loss = self.loss_fn(output_dict, data_dict)
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        score, target = output_dict["score"], data_dict["df_ids"]
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        else:
            acc = (score.max(1)[1] == target).float().mean()
        self.loss_meter.update(loss.item(), data_dict['imgs'].shape[0])
        self.acc_meter.update(acc, 1)

    def do_validate(self):
        logger.info("===> Validation Epoch {} start.".format(self.epoch))
        start_time = time.time()
        self.evaluator.reset()
        self.model.eval()
        for n_iter, data_dict in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                with self.accelerator.autocast():
                    feat = self.model(data)
                    feat = self.accelerator.gather_for_metrics(feat)
                    data = self.accelerator.gather_for_metrics(data)
                    self.evaluator.update((feat, data["df_ids"]))
        cmc, mAP, _, _, _, _ = self.evaluator.compute()
        logger.info("Validation Results - Epoch: {}".format(self.epoch))
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / len(self.val_loader)
        logger.info("===> Validation Epoch {} done. Total time: {:.3f}[s]"
                         .format(self.epoch, end_time - start_time))
        return cmc, mAP
