import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from torch.nn import functional as F
from loss import TripletLoss, CenterLoss, CrossEntropyLabelSmooth
from . import MODEL_FACTORY
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


@MODEL_FACTORY.register_module("resnet50")
class ResNetModel(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(ResNetModel, self).__init__(config, num_classes)
        self.in_planes = 2048
        
        # 使用torchvision的resnet50作为backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 移除原始的全连接层
        del self.resnet.fc

        # 添加分类器
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # 添加瓶颈层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)



    def forward(self, x):
        # 提取特征
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # 全局平均池化
        img_feature = self.resnet.avgpool(x)
        img_feature = torch.flatten(img_feature, 1)
        
        # 瓶颈层处理
        feat = self.bottleneck(img_feature)
        
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, img_feature
        else:
            if self.config["inference"]["neck_feat"] == 'after':
                return feat
            else:
                return img_feature