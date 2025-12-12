import logging
import time
from datetime import timedelta

import torch
from tqdm import tqdm

from loss.supcontrast import SupConLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from . import BaseProcessor
from . import PROCESSOR_FACTORY

logger = logging.getLogger(__name__)

@PROCESSOR_FACTORY.register_module("prompt_learn_caption")
class PromptLearnCaptionProcessor(BaseProcessor):
    def __init__(self, config, model, train_loader_stage1, train_loader_stage2, val_loader,
                 optimizer_stage1, optimizer_stage2,
                 scheduler_stage1, scheduler_stage2, loss_func, num_query, accelerator):
        super(PromptLearnCaptionProcessor, self).__init__(config, model, loss_fn=loss_func,
                                                   accelerator=accelerator, num_query=num_query)
        self.log_period = self.config["train"]["log_period"]
        self.checkpoint_period = self.config["train"]["checkpoint_period"]
        self.eval_period = self.config["train"]["eval_period"]
        self.max_epoch_stage1 = self.config["train"]["max_epoch_stage1"]
        self.max_epoch_stage2 = self.config["train"]["max_epoch_stage2"]

        self.train_loader_stage1 = train_loader_stage1
        self.train_loader_stage2 = train_loader_stage2
        self.val_loader = val_loader
        self.optimizer_stage1 = optimizer_stage1
        self.optimizer_stage2 = optimizer_stage2
        self.scheduler_stage1 = scheduler_stage1
        self.scheduler_stage2 = scheduler_stage2
        self.loss_func = loss_func

        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.caption_loss_meter = AverageMeter()
        self.epoch = self.model.epoch if self.accelerator.num_processes == 1 else self.model.module.epoch
        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=config["test"]["feat_norm"],
                                     reranking=self.config["test"]["re_ranking"])
        logger.info('processor init ...')
        logger.info(
            f'log_period: {self.log_period}, checkpoint_period: {self.checkpoint_period}, eval_period: {self.eval_period}, max_epoch_stage1: {self.max_epoch_stage1}, max_epoch_stage2: {self.max_epoch_stage2}"')

    def do_train_stage1(self):
        logger.info("Start train stage 1 ...")
        image_features = []
        labels = []
        prompts_list = []
        xent = SupConLoss(self.accelerator.device)
        logger.info("Starting to extract image feature ...")
        with torch.no_grad():
            for n_iter, (img, df_id, df_name, img_prompt, img_path) in enumerate(tqdm(self.train_loader_stage1)):
                with self.accelerator.autocast():
                    image_feature = self.model(img, get_image=True)

                    # img_prompt 通常是 tuple (batch_size,), 转为 list
                    batch_prompts = list(img_prompt)

                    for i, img_feat, prompt in zip(df_id, image_feature, batch_prompts):
                        labels.append(i)
                        image_features.append(img_feat.cpu())
                        prompts_list.append(prompt)  # 收集 prompt

            labels_list = torch.stack(labels, dim=0)
            image_features_list = torch.stack(image_features, dim=0)
            # prompt 不需要 stack，保持 list 即可

            batch = self.config['dataset']['train_batch_size']
            num_image = labels_list.shape[0]
            # i_ter = num_image // batch
        del labels, image_features

        self.accelerator.wait_for_everyone()

        for epoch in range(1, self.max_epoch_stage1 + 1):
            logger.info("Epoch[{}]: Start training stage 1 ...".format(epoch))
            self.loss_meter.reset()
            self.model.train()
            self.scheduler_stage1.step(epoch)
            iter_list = torch.randperm(num_image)
            batches = torch.split(iter_list, batch)
            for i, b_list in enumerate(tqdm(batches)):
                # train step
                self.optimizer_stage1.zero_grad()

                indices = b_list.to(self.accelerator.device)
                indices_cpu = b_list.tolist()

                target = labels_list[indices].to(self.accelerator.device)
                batch_image_features = image_features_list[indices_cpu].to(self.accelerator.device)
                batch_prompts = [prompts_list[idx] for idx in indices_cpu]

                with self.accelerator.autocast():
                    # 关键修改：传入 batch_prompts
                    text_features = self.model(label=target, get_text=True)

                loss_i2t = xent(batch_image_features, text_features, target, target)
                loss_t2i = xent(text_features, batch_image_features, target, target)
                loss = loss_i2t + loss_t2i

                self.accelerator.backward(loss)
                self.optimizer_stage1.step()
                self.loss_meter.update(loss.item(), len(b_list))

                if (i + 1) % self.log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                     .format(epoch, (i + 1), len(self.train_loader_stage1),
                                             self.loss_meter.avg, self.scheduler_stage1._get_lr(epoch)[0]))

            if epoch % self.checkpoint_period == 0:
                if self.accelerator.is_main_process:
                    self.save_state()

    def do_train_stage2(self):
        logger.info("Start train stage 2 ...")
        best_metrics = {"mAP": 0.0, "Rank@1": 0.0, "Rank@5": 0.0, "Rank@10": 0.0, "Epoch": 0}
        # train
        batch = self.config['dataset']['train_batch_size']
        i_ter = self.config['dataset']['num_classes'] // batch
        left = self.config['dataset']['num_classes'] - batch * (self.config['dataset']['num_classes'] // batch)
        if left != 0:
            i_ter = i_ter + 1
        text_features = []
        logger.info("Starting to extract text feature ...")
        with torch.no_grad():
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, self.config['dataset']['num_classes'])
                with self.accelerator.autocast():
                    text_feature = self.model(label=l_list, get_text=True)
                text_features.append(text_feature.cpu())
            text_features = torch.cat(text_features, 0).to(self.accelerator.device)

        logger.info("do valid!")
        cmc, mAP = self.do_validate()
        best_metrics = {"mAP": mAP, "Rank@1": cmc[0], "Rank@5": cmc[4], "Rank@10": cmc[9], "Epoch": 0}
        logger.info(
            "===> Best mAP: {:.3f}, Rank@1: {:.3f}, Rank@5: {:.3f}, Rank@10: {:.3f}, Epoch: {}".format(
                best_metrics["mAP"], best_metrics["Rank@1"], best_metrics["Rank@5"],
                best_metrics["Rank@10"], best_metrics["Epoch"]))

        logger.info("Starting to train image encoder")
        self.model.train()
        all_start_time = time.monotonic()
        for epoch in range(1, self.max_epoch_stage2 + 1):
            start_time = time.time()
            self.loss_meter.reset()
            self.acc_meter.reset()
            self.evaluator.reset()
            self.model.train()
            for n_iter, (img, df_id, df_name, img_prompt, img_path) in enumerate(tqdm(self.train_loader_stage2)):
                self.optimizer_stage2.zero_grad()
                batch_prompts = list(img_prompt)
                with (self.accelerator.autocast()):
                    output_dict = self.model(x=img, label=df_id, captions=batch_prompts)

                    logits = output_dict['image_features'] @ text_features.t()
                    loss = self.loss_fn(output_dict, df_id, logits)

                self.accelerator.backward(loss)
                self.optimizer_stage2.step()
                self.scheduler_stage2.step()

                acc = (logits.max(1)[1] == df_id).float().mean()

                self.loss_meter.update(loss.item(), img.shape[0])
                self.acc_meter.update(acc, 1)

                if (n_iter + 1) % self.log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                     .format(epoch, (n_iter + 1), len(self.train_loader_stage2),
                                             self.loss_meter.avg, self.acc_meter.avg, self.scheduler_stage2.get_lr()[0]))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                             .format(epoch, time_per_batch,
                                     self.config["dataset"]["train_batch_size"] / time_per_batch))

            if epoch % self.checkpoint_period == 0:
                if self.accelerator.is_main_process:
                    self.save_state()
            if epoch % self.eval_period == 0 or epoch == self.max_epoch_stage2 - 1:
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

    def do_validate(self):
        logger.info("===> Validation Epoch {} start.".format(self.epoch))
        start_time = time.time()
        self.evaluator.reset()
        self.model.eval()
        for n_iter, (img, df_id, df_name, img_prompt, img_path) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                feat = self.model(img)
                feat, df_id = self.accelerator.gather_for_metrics((feat, df_id))
                self.evaluator.update((feat, df_id))
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
