import glob
import json
import logging
import re

import os.path as osp

from datasets import DATASETS_FACTORY
from .base_dataset import BaseImageDataset
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

@DATASETS_FACTORY.register_module("DFT30")
class DFT30(BaseImageDataset):
    def __init__(self, config):
        super(DFT30, self).__init__(config)
        self.config = config
        self.root_dir = config.get('root_dir')
        self.dataset_name = config.get('dataset_name')
        self.dataset_json_file = config.get('dataset_json_file')
        print(self.dataset_json_file)
        with open(self.dataset_json_file , 'r') as f:
            dataset_json = json.load(f)
        self.train_dataset_info = dataset_json['train']
        self.query_dataset_info = dataset_json['query']
        self.gallery_dataset_info = dataset_json['gallery']
        self.dataset_dir = osp.join(self.root_dir, self.dataset_name)
        self.df_id_begin = 0
        self.df_id_name_map = {}
        self.df_name_id_map = {}
        train = self._process_dir(self.dataset_dir, self.train_dataset_info, relabel=True, train=True)
        query = self._process_dir(self.dataset_dir, self.query_dataset_info, relabel=False, query=True)
        gallery = self._process_dir(self.dataset_dir, self.gallery_dataset_info, relabel=False, gallery=True)

        if self.config.get('verbose', False):
            logger.info("=> DFT30 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_df_ids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_query_df_ids, self.num_query_imgs = self.get_imagedata_info(self.query)
        self.num_gallery_df_ids, self.num_gallery_imgs= self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, dataset_info, relabel=False, train=False, query=False, gallery=False):
        dataset = []
        for df_name in dataset_info:
            if train:
                for img_info in dataset_info[df_name]:
                    img_path = osp.join(dir_path, df_name, img_info['image'])
                    img_prompt = img_info['prompt']
                    if df_name not in self.df_name_id_map:
                        self.df_id_name_map[self.df_id_begin] = df_name
                        self.df_name_id_map[df_name] = self.df_id_begin
                        self.df_id_begin += 1
                    df_id = self.df_name_id_map[df_name]
                    dataset.append((img_path, df_id, img_prompt))
            elif query:
                for img_info in dataset_info[df_name]:
                    img_path = osp.join(dir_path, df_name, img_info['image'])
                    img_prompt = img_info['prompt']
                    if df_name not in self.df_name_id_map:
                        self.df_id_name_map[self.df_id_begin] = df_name
                        self.df_name_id_map[df_name] = self.df_id_begin
                        self.df_id_begin += 1
                    df_id = self.df_name_id_map[df_name]
                    dataset.append((img_path, df_id, img_prompt))
            elif gallery:
                for img_info in dataset_info[df_name]:
                    img_path = osp.join(dir_path, df_name, img_info['image'])
                    img_prompt = img_info['prompt']
                    if df_name not in self.df_name_id_map:
                        self.df_id_name_map[self.df_id_begin] = df_name
                        self.df_name_id_map[df_name] = self.df_id_begin
                        self.df_id_begin += 1
                    df_id = self.df_name_id_map[df_name]
                    dataset.append((img_path, df_id, img_prompt))
        return dataset