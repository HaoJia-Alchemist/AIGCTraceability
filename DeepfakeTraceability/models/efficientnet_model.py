import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

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

@MODEL_FACTORY.register_module("efficientnet-b4")
class EfficientNetModel(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(EfficientNetModel, self).__init__(config, num_classes)
        self.in_planes = 1792
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', weights_path=self.config[
            'pretrained'])

        del self.efficientnet._fc

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        img_feature = self.efficientnet.extract_features(x)
        img_feature = self.efficientnet._avg_pooling(img_feature)
        img_feature = img_feature.flatten(start_dim=1)
        feat = self.bottleneck(img_feature)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, img_feature

        else:
            if self.config["inference"]["neck_feat"] == 'after':
                return feat
            else:
                return img_feature