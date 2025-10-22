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
        accelerator = Accelerator()
        device = accelerator.device
    except:
        accelerator = None
        device = "cpu"
    set_seed(config.get('seed', 1234))

    # 创建日志
    log_file = os.path.join(os.path.dirname(config["model_weights"]), "test.log")
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
    model = make_model(config['model'], num_classes=num_classes)
    model.to(device)

    model = accelerator.prepare(model)

    # prepare the processor
    processor_class = PROCESSOR_FACTORY[config['test']['processor']]
    processor = processor_class(config, model, val_loader=val_loader, num_query=num_query, accelerator=accelerator)
    processor.load_model(config["model_weights"])
    cmc ,mAP = processor.do_validate()
    best_metrics = {"mAP": mAP, "Rank@1": cmc[0], "Rank@5": cmc[4], "Rank@10": cmc[9]}
    logger.info("Stop Testing on best Testing metric \n{}".format(parse_metric_for_print(best_metrics)))


if __name__ == '__main__':
    main()
