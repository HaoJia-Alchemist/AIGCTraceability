import numpy as np
import torch
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import albumentations as A
import torchvision.transforms as T
import cv2
from datasets.albu import IsotropicResize

ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of dft dataset
    """

    def get_imagedata_info(self, data):
        df_ids = []
        for _, df_id, _ in data:
            df_ids += [df_id]
        df_ids = set(df_ids)
        num_df_ids = len(df_ids)
        num_imgs = len(data)
        return num_df_ids, num_imgs

    def print_dataset_statistics(self, train, query, gallery):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image dft dataset
    """
    def __init__(self, config):
        self.config = config

    def get_data_augmentation(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5),
        ])
        return trans
    def get_data_transforms(self):
        trans = T.Compose([
            T.Resize(self.config['resolution']),
            T.ToTensor(),
            T.Normalize(mean=self.config['mean'], std=self.config['std'])
        ])
        return trans

    def print_dataset_statistics(self, train, query, gallery):
        num_train_df_ids, num_train_imgs = self.get_imagedata_info(train)
        num_query_df_ids, num_query_imgs = self.get_imagedata_info(query)
        num_gallery_df_ids, num_gallery_imgs = self.get_imagedata_info(gallery)

        logger.info("Dataset statistics:")
        logger.info("  ------------------------------------")
        logger.info("  subset   | # df_ids | # df_images ")
        logger.info("  ------------------------------------")
        logger.info("  train    | {:5d}    | {:8d} ".format(num_train_df_ids, num_train_imgs))
        logger.info("  query    | {:5d}    | {:8d} ".format(num_query_df_ids, num_query_imgs))
        logger.info("  gallery  | {:5d}    | {:8d} ".format(num_gallery_df_ids, num_gallery_imgs))
        logger.info("  ------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, data_augmentation=None, df_id_name_map=None, df_name_id_map=None):
        self.dataset = dataset
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.df_id_name_map = df_id_name_map
        self.df_name_id_map = df_name_id_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, df_id, img_prompt = self.dataset[index]
        df_name = self.df_id_name_map[df_id]
        img = read_image(img_path)
        if self.data_augmentation is not None:
            img = np.array(img, dtype=np.uint8)
            transformed = self.data_augmentation(image=img)
            img = transformed['image']
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, df_id, df_name, img_prompt, img_path.split('/')[-1]