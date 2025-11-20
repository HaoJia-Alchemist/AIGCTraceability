import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf

from datasets import make_dataloader
from models import make_model
from processor import PROCESSOR_FACTORY
from solver.cosine_lr import CosineLRScheduler
from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import create_logger
from utils.metrics import parse_metric_for_print


# from DeepfakeTraceability.datasets import FalDataset
# from DeepfakeTraceability.datasets import GANBlendingDataset
# from DeepfakeTraceability.datasets import NaCLDataset
# from DeepfakeTraceability.datasets import RegeneratedDataset
# from detectors import DETECTOR
# from logger import create_logger, RankFilter
# from metrics.utils import parse_metric_for_print
# from optimizor.LinearLR import LinearDecayLR
# from processor import TRAINER


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--config_file', type=str,
                        default='./config/configs_dir/efficientnet.yaml',
                        help='path to detector YAML file')
    parser.add_argument("--opts", nargs='+', help="Modify config options using the command-line", default=[])
    args = parser.parse_args()

    # parse options and load config
    config = OmegaConf.load(args.config_file)
    config_train = OmegaConf.load('config/train_config.yaml')
    config = OmegaConf.merge(config_train, config)
    config_args = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(config, config_args)
    # 初始化Accelerator（用于获取进程信息）
    try:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        device = accelerator.device
    except:
        accelerator = None
        device = "cpu"
    set_seed(config.get('seed', 1234))

    # 创建日志
    log_file = os.path.join(config.get('log_dir', 'logs'),
                            f"{config.get('task_name', '')}_{time.strftime('%Y%m%d%H%M%S')}", "train.log")
    config["logging"]["log_file"] = log_file
    config["logging"]["log_dir"] = os.path.dirname(log_file)
    logger = create_logger(config['logging'])

    # 展示配置文件
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # prepare the training data loader
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes = make_dataloader(config['dataset'])

    # prepare the model
    model = make_model(config, num_classes=num_classes)
    model = model.float()
    # prepare the loss
    loss_func, center_criterion = model.make_loss(config, num_classes=num_classes)

    # prepare the optimizer
    optimizer_stage1 = model.make_optimizer_stage1(config, model)

    # prepare the scheduler
    scheduler_stage1 = lr_scheduler = CosineLRScheduler(
        optimizer_stage1,
        t_initial=config['train']['max_epoch_stage1'],
        lr_min=config['solver']['stage1']['lr_min'],
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=config['solver']['stage1']['warmup_lr_init'],
        warmup_t=config['solver']['stage1']['warmup_epoch'],
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )




    optimizer_stage2, optimizer_center_stage2 = model.make_optimizer_stage2(config, model, center_criterion)
    scheduler_stage2 = WarmupMultiStepLR(optimizer_stage2, config['solver']['stage2']['steps'],
                                         config['solver']['stage2']['gamma'],
                                         config['solver']['stage2']['warmup_factor'],
                                         config['solver']['stage2']['warmup_iters'],
                                         config['solver']['stage2']['warmup_method'])

    model, center_criterion, optimizer_stage1, optimizer_stage2, optimizer_center_stage2 = accelerator.prepare(model,
                                                                                                               center_criterion,
                                                                                                               optimizer_stage1,
                                                                                                               optimizer_stage2,
                                                                                                               optimizer_center_stage2)

    accelerator.register_for_checkpointing(scheduler_stage1)
    accelerator.register_for_checkpointing(scheduler_stage2)

    # prepare the processor
    processor_class = PROCESSOR_FACTORY[config['train']['processor']]
    processor = processor_class(config, model, center_criterion, train_loader_stage1, train_loader_stage2, val_loader,
                                optimizer_stage1, optimizer_stage2, optimizer_center_stage2,
                                scheduler_stage1, scheduler_stage2, loss_func, num_query, accelerator)

    if config["resume"]:
        processor.load_state(config["resume"])
        processor.epoch = config["resume_epoch"]
        logger.info("Resume from checkpoint: {}".format(config["resume"]))

    processor.do_train_stage1()
    best_metrics = processor.do_train_stage2()


    logger.info("Stop Training on best Testing metric \n{}".format(parse_metric_for_print(best_metrics)))


if __name__ == '__main__':
    main()
