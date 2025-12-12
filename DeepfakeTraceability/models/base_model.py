import logging
import torch
from torch import nn
from torch.nn import functional as F
from loss import CenterLoss, TripletLoss, CrossEntropyLabelSmooth
import os.path as osp

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __init__(self, config=None, num_classes=10):
        super(BaseModel, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.epoch = 0

    def make_loss(self, config, num_classes=10):
        sampler = config['dataset']['sampler']
        if 'triplet' in config['model']['metric_loss_type']:
            if config['model']['no_margin']:
                triplet = TripletLoss()
                logger.info("using soft triplet loss for training")
            else:
                triplet = TripletLoss(config['solver']['margin'])  # triplet loss
                logger.info("using triplet loss with margin:{}".format(config['solver']['margin']))
        else:
            logger.error('expected METRIC_LOSS_TYPE should be triplet'
                         'but got {}'.format(config['model']['metric_loss_type']))

        if config['model']['if_label_smooth'] == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
            logger.info("label smooth on, num_classes: {}".format(num_classes))

        if sampler == 'softmax':
            def loss_func(output_dict, data_dict):
                cls_score = output_dict['cls_score']
                target = data_dict['df_ids']
                return F.cross_entropy(cls_score, target)

        elif sampler == 'softmax_triplet':
            # def loss_func(score, feat, target, i2tscore=None):
            def loss_func(output_dict, data_dict, i2tscore=None):
                cls_score = output_dict['cls_score']
                target = data_dict['df_ids']
                image_features = output_dict['image_features']
                if config['model']['metric_loss_type'] == 'triplet':
                    if config['model']['if_label_smooth'] == 'on':
                        if isinstance(cls_score, list):
                            ID_LOSS = [xent(scor, target) for scor in cls_score[0:]]
                            ID_LOSS = sum(ID_LOSS)
                        else:
                            ID_LOSS = xent(cls_score, target)

                        if isinstance(image_features, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in image_features[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                        else:
                            TRI_LOSS = triplet(image_features, target)[0]

                        loss = config['model']['id_loss_weight'] * ID_LOSS + config['model'][
                            'triplet_loss_weight'] * TRI_LOSS

                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = config['model']['i2t_loss_weight'] * I2TLOSS + loss

                        return loss
                    else:
                        if isinstance(cls_score, list):
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in cls_score[0:]]
                            ID_LOSS = sum(ID_LOSS)
                        else:
                            ID_LOSS = F.cross_entropy(cls_score, target)

                        if isinstance(image_features, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in image_features[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                        else:
                            TRI_LOSS = triplet(image_features, target)[0]

                        loss = config['model']['id_loss_weight'] * ID_LOSS + config['model'][
                            'triplet_loss_weight'] * TRI_LOSS

                        if i2tscore != None:
                            I2TLOSS = F.cross_entropy(i2tscore, target)
                            loss = config['model']['i2t_loss_weight'] * I2TLOSS + loss

                        return loss
                else:
                    logger.error('expected METRIC_LOSS_TYPE should be triplet'
                                 'but got {}'.format(config['model']['metric_loss_type']))

        else:
            logger.error('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
                         'but got {}'.format(config['dataset']['sampler']))

        return loss_func

    def make_optimizer(self, config, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = config["solver"]['base_lr']
            weight_decay = config["solver"]['weight_decay']
            if "bias" in key:
                lr = config["solver"]['base_lr'] * config["solver"]['bias_lr_factor']
                weight_decay = config["solver"]['weight_decay_bias']
            if config["solver"]['large_fc_lr']:
                if "classifier" in key or "arcface" in key:
                    lr = config["solver"]['base_lr'] * 2
                    logger.info('Using two times learning rate for fc ')

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if config["solver"]['type'] == 'SGD':
            optimizer = getattr(torch.optim, config["solver"]['type'])(params, momentum=config["solver"]['momentum'])
        elif config["solver"]['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=config["solver"]['base_lr'],
                                          weight_decay=config["solver"]['weight_decay'])
        else:
            optimizer = getattr(torch.optim, config["solver"]['type'])(params)
        return optimizer

    def save(self, save_path):
        torch.save(self.state_dict(), osp.join(self.config["logging"]["log_dir"], "best_model", "model.pth"))

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location="cpu"))
