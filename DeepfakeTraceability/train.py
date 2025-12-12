import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from accelerate import Accelerator
from omegaconf import OmegaConf

from datasets import make_dataloader
from loss import make_loss
from models import make_model
from processor import PROCESSOR_FACTORY
from solver import make_optimizer
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
    parser.add_argument('--train_config_file', type=str,
                        default='./config/train_config.yaml',
                        help='path to train YAML file')
    parser.add_argument("--opts", nargs='+', help="Modify config options using the command-line", default=[])
    args = parser.parse_args()

    # parse options and load config
    config = OmegaConf.load(args.config_file)
    config_train = OmegaConf.load(args.train_config_file)
    config = OmegaConf.merge(config_train, config)
    config_args = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(config, config_args)

    # 初始化Accelerator（用于获取进程信息）
    try:
        accelerator = Accelerator()
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
    train_loader, train_loader_normal, val_loader, num_query, num_classes = make_dataloader(config['dataset'])

    # prepare the model
    model = make_model(config, num_classes=num_classes)
    model = model.float()
    model.to(device)
    # prepare the loss
    loss_func= model.make_loss(config, num_classes=num_classes)

    # prepare the optimizer
    optimizer = model.make_optimizer(config, model)

    # prepare the scheduler
    scheduler = WarmupMultiStepLR(optimizer, config['solver']['steps'], config['solver']['gamma'],
                                  config['solver']['warmup_factor'],
                                  config['solver']['warmup_iters'], config['solver']['warmup_method'])

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)

    # prepare the processor
    processor_class = PROCESSOR_FACTORY[config['train']['processor']]
    processor = processor_class(config, model, train_loader, val_loader, optimizer,
                                scheduler, loss_func, num_query, accelerator)

    if config["resume"]:
        processor.load_state(config["resume"])
        processor.epoch = config["resume_epoch"]

        logger.info("Resume from checkpoint: {}".format(config["resume"]))

    best_metrics = processor.do_train()
    logger.info("Stop Training on best Testing metric \n{}".format(parse_metric_for_print(best_metrics)))

if __name__ == '__main__':
    main()
