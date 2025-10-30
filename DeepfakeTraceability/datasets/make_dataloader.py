from accelerate import Accelerator, DataLoaderConfiguration

from datasets import DATASETS_FACTORY
import torch
from torch.utils.data import DataLoader
from datasets.base_dataset import ImageDataset, BaseImageDataset
from .sampler import RandomDeepfakeSampler
import logging
# 创建日志记录器
logger = logging.getLogger('Train')

def collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, df_ids, df_names, img_prompts, img_names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    df_ids = torch.tensor(df_ids, dtype=torch.int64)
    df_names = list(df_names)
    img_prompts = list(img_prompts)
    img_names = list(img_names)
    return imgs, df_ids, df_names, img_prompts, img_names

def make_dataloader(config):
    accelerator = Accelerator()
    dataset = DATASETS_FACTORY[config['dataset_name']](config)
    num_workers = config['num_workers']
    data_augmentation = dataset.get_data_augmentation() if config['use_data_augmentation'] else None
    data_transforms = dataset.get_data_transforms()
    train_set = ImageDataset(dataset.train, data_transforms, data_augmentation, dataset.df_id_name_map, dataset.df_name_id_map)
    train_set_normal = ImageDataset(dataset.train, data_transforms, None, dataset.df_id_name_map, dataset.df_name_id_map)
    num_classes = dataset.num_train_df_ids
    val_set = ImageDataset(dataset.query + dataset.gallery, data_transforms, None, dataset.df_id_name_map, dataset.df_name_id_map)

    if 'triplet' in config['sampler']:
        train_loader = DataLoader(
                train_set, batch_size=config['train_batch_size'],
                sampler=RandomDeepfakeSampler(dataset.train, batch_size=config['train_batch_size'], num_instances=config['num_instances'], accelerator=accelerator),
                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
            )

    elif config['sampler'] == 'softmax':
        logger.info('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=config['train_batch_size'], shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        logger.info('unsupported sampler! expected softmax or triplet but got {}'.format(config.SAMPLER))

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=config['train_batch_size'], shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=config['test_batch_size'], shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    train_loader = accelerator.prepare(train_loader)
    train_loader_normal = accelerator.prepare(train_loader_normal)
    val_loader = accelerator.prepare(val_loader)
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes
