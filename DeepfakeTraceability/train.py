import argparse
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler

from DeepfakeTraceability.datasets import FalDataset
from DeepfakeTraceability.datasets import GANBlendingDataset
from DeepfakeTraceability.datasets import NaCLDataset
from DeepfakeTraceability.datasets import RegeneratedDataset
from detectors import DETECTOR
from logger import create_logger, RankFilter
from metrics.utils import parse_metric_for_print
from optimizor.LinearLR import LinearDecayLR
from trainer import TRAINER

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--config_file', type=str,
                    default='/home/jh/disk/workspace/DeepfakeBenchV2/training/config/detector/fal_xception.yaml',
                    help='path to detector YAML file')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("--opts", nargs='+', help="Modify config options using the command-line", default=[])
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])
        np.random.seed(config['manualSeed'])
        random.seed(config['manualSeed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def prepare_training_data(config):
    # Only use the blending datasets class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model']['name'] == 'facexray':
            train_set = FFBlendDataset(config, mode='train')
        elif config['model']['name'] == 'fwa':
            train_set = FWABlendDataset(config, mode='train')
        elif config['model']['name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model']['name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        elif config['model']['name'] == 'yzy':
            train_set = YZYDataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending datasets'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'shuffle':
        train_set = ShuffleDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair datasets class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'fal':
        train_set = FalDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'nacl_dataset':
        train_set = NaCLDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'regenerated_dataset':
        train_set = RegeneratedDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'ganblending_dataset':
        train_set = GANBlendingDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='train',
        )
    if config['model']['name'] == 'lsda':
        from DeepfakeTraceability.datasets import CustomSampler
        custom_sampler = CustomSampler(num_groups=2 * 360, n_frame_per_vid=config['frame_num']['train'],
                                       batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler,
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
            )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing datasets
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test datasets
        if config.get('dataset_type', None) == 'lrl':
            test_set = LRLDataset(
                config=config,
                mode='test',
            )
        elif config.get('dataset_type', None) == 'fal':
            test_set = FalDataset(
                config=config,
                mode='test',
            )
        # elif config.get('dataset_type', None) == 'regenerated_dataset':
        #     test_set = regenerated_dataset(
        #         config=config,
        #         mode='test',
        #     )
        else:
            test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                # drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    return model.get_optimizer(config)


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs'] / 4),
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    config = OmegaConf.load(args.config_file)
    config_train = OmegaConf.load('config/train_config.yaml')
    config = OmegaConf.merge(config, config_train)
    config_args = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(config, config_args)
    config['local_rank'] = args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False
    config['log_dir'] = os.path.join(config['log_dir'], f"{config['task_name']}_{time.strftime('%Y%m%d%H%M%S')}")
    logger_path = config['log_dir']
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Config file path {}'.format(args.config_file))
    logger.info('Save log to {}'.format(logger_path))
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=60)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model']['name']]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)
    print(optimizer)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the scale
    scaler = GradScaler()

    if config['checkpoints']:
        ckpt = torch.load(config['checkpoints'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=True)
        optimizers_checkpoints = ckpt['optimizer']
        if isinstance(ckpt['optimizer'], tuple):
            for idx, optimizers_item in enumerate(optimizers_checkpoints):
                optimizer[idx].load_state_dict(optimizers_item)
                for state in optimizer[idx].state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        else:
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        scaler.load_state_dict(ckpt['scaler'])
        config['start_epoch'] = ckpt['epoch'] + 1
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the checkpoint')

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer_class = TRAINER[config['trainer']]
    trainer = trainer_class(config, model, optimizer, scheduler, logger, metric_scoring, scaler)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric)))
    # update
    if 'svdd' in config['model']['name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()