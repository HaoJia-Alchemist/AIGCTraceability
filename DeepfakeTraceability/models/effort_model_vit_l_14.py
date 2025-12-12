import logging

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

logger = logging.getLogger(__name__)

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
        self.token_embedding = clip_model.token_embedding
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

    def encode_caption(self, text):
        # 新增：处理真实的 Caption 文本
        # text: [batch_size, 77] (tokenized indices)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        # 添加位置编码 (需截断以匹配长度，标准CLIP通常是77)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # 取 EOT (End of Text) 特征
        # text.argmax(dim=-1) 找到 EOT token 的位置
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

@MODEL_FACTORY.register_module(module_name='effort_vit_l_14')
class EffortViTL14Detector(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(EffortViTL14Detector, self).__init__(config, num_classes)
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

    def forward(self, x=None, label=None, captions=None, get_image=False, get_text=False):

        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
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
        logger.info('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        logger.info('Loading pretrained model for finetuning from {}'.format(model_path))


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

    def make_optimizer_stage2(self, config, model):
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
                    logger.info('Using two times learning rate for fc ')


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
        return optimizer


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
    def __init__(self, num_classes, class_names=None, clip_model=None):
        super().__init__()
        # 定义初始化模板
        # X 代表我们要学习的 Context Vector，CLASS 代表类别名称插入位置
        # 例如: "A photo generated by X X X X [CLASS]"
        n_ctx = 16  # 学习 16 个 Context Token
        prompt_prefix = " ".join(["X"] * n_ctx) + " CLASS"

        # 获取 CLIP 的 Token Embedding 层
        token_embedding = clip_model.token_embedding
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names
        self.num_classes = num_classes

        # 1. 初始化 Learnable Context Vectors (Vectors for "X")
        # 使用 "a photo of a" 的 embedding 进行初始化通常比随机初始化收敛更快
        logger.info(f"Initializing Prompt Learner with {n_ctx} context vectors.")
        ctx_vectors = torch.empty(num_classes, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)

        # 注册为 Parameter
        self.cls_ctx = nn.Parameter(ctx_vectors)  # shape: [n_ctx, dim]

        # 2. 准备类别名称的 Token Embeddings
        # 比如 ["Midjourney", "DALL-E", ...]
        # 如果没有具体的名称，可以使用 ["Generator 0", "Generator 1", ...]
        class_name_lens = []
        logger.info(f"Number of context words (tokens): {n_ctx}")
        prompts = []
        for name in class_names:
            name = name.replace("_", " ")
            prompts.append(prompt_prefix + " " + name + ".")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, 77]
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        # [SOS] - 取第一个类别的即可，因为所有 SOS 都一样
        # 保存为 [1, 1, dim] 以便后续 expand
        self.register_buffer("token_prefix", embedding[:1, :1, :])

        # [Class Name + EOS + Padding] - 每个类别不同
        # 保存为 [n_cls, *, dim]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, label):
            """
            Args:
                label (torch.Tensor): 类别索引，形状为 [Batch] 或 [Num_Classes]
            Returns:
                prompts (torch.Tensor): 形状为 [Batch, 77, dim]
            """
            # 1. 获取对应类别的 Context Vector
            cls_ctx = self.cls_ctx[label]  # (B, n_ctx, dim)
            b = label.shape[0]

            # 2. 扩展 Prefix
            prefix = self.token_prefix.expand(b, -1, -1)  # (B, 1, dim)

            # 3. 获取对应类别的 Suffix (包含 Class Name)
            # 注意：这里必须索引，因为每个类的 Class Name 不同
            suffix = self.token_suffix[label]  # (B, *, dim)

            # 4. 拼接
            prompts = torch.cat(
                [
                    prefix,  # (B, 1, dim)
                    cls_ctx,  # (B, n_ctx, dim)
                    suffix,  # (B, *, dim)
                ],
                dim=1,
            )

            return prompts