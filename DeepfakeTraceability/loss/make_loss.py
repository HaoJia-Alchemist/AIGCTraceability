import logging

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
logger = logging.getLogger(__name__)
def make_loss(config, num_classes=10):
    sampler = config['dataset']['sampler']
    feat_dim = config['model']['feat_dim']
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)  # center loss
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
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, i2tscore=None):
            if config['model']['metric_loss_type'] == 'triplet':
                if config['model']['if_label_smooth'] == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = config['model']['id_loss_weight'] * ID_LOSS + config['model']['triplet_loss_weight'] * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = config['model']['i2t_loss_weight'] * I2TLOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = config['model']['id_loss_weight'] * ID_LOSS + config['model']['triplet_loss_weight'] * TRI_LOSS

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
    return loss_func, center_criterion