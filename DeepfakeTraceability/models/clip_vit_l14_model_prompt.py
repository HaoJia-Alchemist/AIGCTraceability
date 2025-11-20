from accelerate import accelerator

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn

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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding #.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) #.type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

@MODEL_FACTORY.register_module(module_name='CLIP_ViT_L14_Model_Prompt')
class CLIP_ViT_L14_Model_Prompt(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(CLIP_ViT_L14_Model_Prompt, self).__init__(config, num_classes)
        self.config = config
        self.epoch = 0
        self.num_classes = num_classes
        self.in_planes = 1024

        self.backbone_name = config['model']['backbone']
        self.neck_feat = config['test']['neck_feat']
        if self.backbone_name == 'ViT-B/16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.backbone_name == 'ViT-L/14':
            self.in_planes = 1024
            self.in_planes_proj = 768
        elif self.backbone_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((config['dataset']['resolution'] - config['model']['stride_size']) // config['model']['stride_size'] + 1)
        self.w_resolution = int((config['dataset']['resolution'] - config['model']['stride_size']) // config['model']['stride_size'] + 1)
        self.vision_stride_size = config['model']['stride_size']
        clip_model = load_clip_to_cpu(self.backbone_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.image_encoder = clip_model.visual

        dataset_name = config['dataset']['name']
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, label=None, get_image=False, get_text=False):
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.backbone_name == 'RN50':
                return image_features_proj[0]
            elif self.backbone_name == 'ViT-L/14':
                return image_features_proj[:, 0]

        if self.backbone_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.backbone_name == 'ViT-L/14':
            cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


    def make_optimizer_stage1(self, config, model):
        params = []
        keys = []
        for key, value in model.named_parameters():
            if "prompt_learner" in key:
                lr = config['solver']['stage1']['base_lr']
                weight_decay = config['solver']['stage1']['weight_decay']
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                keys += [key]
        if config['solver']['stage1']['type'] == 'SGD':
            optimizer = getattr(torch.optim, config['solver']['stage1']['type'])(params,
                                                                               momentum=config['solver']['stage1']['momentum'])
        elif config['solver']['stage1']['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=config['solver']['stage1']['base_lr'],
                                          weight_decay=config['solver']['stage1']['weight_decay'])
        else:
            optimizer = getattr(torch.optim, config['solver']['stage1']['type'])(params)
        return optimizer

    def make_optimizer_stage2(self, config, model, center_criterion):
        params = []
        keys = []
        for key, value in model.named_parameters():
            if "text_encoder" in key:
                # value.requires_grad_(False)
                continue
            if "prompt_learner" in key:
                # value.requires_grad_(False)
                continue
            if not value.requires_grad:
                continue
            lr = config['solver']['stage2']['base_lr']
            weight_decay = config['solver']['stage2']['weight_decay']
            if "bias" in key:
                lr = config['solver']['stage2']['base_lr'] * config['solver']['stage2']['bias_lr_factor']
                weight_decay = config['solver']['stage2']['weight_decay_bias']
            if config['solver']['stage2']['large_fc_lr']:
                if "classifier" in key or "arcface" in key:
                    lr = config['solver']['stage2']['base_lr'] * 2
                    print('Using two times learning rate for fc ')


            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
        if config['solver']['stage2']['type'] == 'SGD':
            optimizer = getattr(torch.optim, config['solver']['stage2']['type'])(params,
                                                                               momentum=config['solver']['stage2']['momentum'])
        elif config['solver']['stage2']['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=config['solver']['stage2']['base_lr'],
                                          weight_decay=config['solver']['stage2']['weight_decay'])
        else:
            optimizer = getattr(torch.optim, config['solver']['stage2']['type'])(params)
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=config['solver']['stage2']['center_lr'])

        return optimizer, optimizer_center


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, "/home/jh/disk/pretrained/clip/")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 768
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        tokenized_prompts = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim,)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts