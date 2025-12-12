import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import CenterLoss, TripletLoss, CrossEntropyLabelSmooth
from . import MODEL_FACTORY

from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

from .base_model import BaseModel


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

@MODEL_FACTORY.register_module(module_name='clip_vit_l14')
class CLIP_ViT_L14_Model(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(CLIP_ViT_L14_Model, self).__init__(config, num_classes)
        self.config = config
        self.epoch = 0
        self.num_classes = num_classes
        self.in_planes = 1024

        self.backbone = self.build_backbone(config)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


    def build_backbone(self, config):
        # ⚠⚠⚠ Download CLIP model using the below link
        # https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link

        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]

        # ViT-L/14 224*224
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14")  # the path of this folder in your disk (download from the above link)

        return clip_model.vision_model

    def features(self, x) -> torch.tensor:
        feat = self.backbone(x)['pooler_output']
        return feat

    def forward(self, x) -> dict:
        img_feature = self.features(x)
        feat = self.bottleneck(img_feature)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, img_feature

        else:
            if self.config["test"]["neck_feat"] == 'after':
                return feat
            else:
                return img_feature

class Adapter(nn.Module):
    """
    一个简单的残差 MLP 结构：
    Input -> Linear(降维) -> ReLU -> Linear(升维) -> Output
    """
    def __init__(self, channels, reduction=4, ratio=0.2):
        super().__init__()
        # 降维后的维度 (Bottleneck)
        self.reduced_channels = channels // reduction
        self.ratio = ratio # 残差融合比例，论文推荐 0.2 (20% 新特征 + 80% 原特征)

        self.adapter = nn.Sequential(
            nn.Linear(channels, self.reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, channels, bias=False)
        )

    def forward(self, x):
        # x 是原始 CLIP 的特征向量
        residual = self.adapter(x)
        # 融合：原始特征 + ratio * 新学到的特征
        return x + self.ratio * residual