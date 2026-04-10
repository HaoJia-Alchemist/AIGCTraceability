import math
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import os
import os.path as osp
from loss import CenterLoss, CrossEntropyLabelSmooth, TripletLoss
from . import MODEL_FACTORY
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

@MODEL_FACTORY.register_module(module_name='lora_coop_decouple')
class LoraCoopDecoupleDetector(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(LoraCoopDecoupleDetector, self).__init__(config, num_classes)
        model_name = "openai/clip-vit-large-patch14"
        self.in_planes = 1024
        self.in_planes_proj = 768

        clip_model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.content_adapter = ResidualAdapter(self.in_planes_proj, self.in_planes_proj)

        self.artifact_adapter = ResidualAdapter(self.in_planes, self.in_planes)

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

        self.image_encoder = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        peft_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none"
        )

        # 将 Vision Model 包装为 PEFT Model
        # 这一步会自动冻结原始权重，只开启 LoRA 参数的梯度
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)
        self.image_encoder.print_trainable_parameters()

        self.prompt_learner = PromptLearner(num_classes, clip_model=clip_model)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, data_dict=None, label=None, captions=None, get_image=False, get_text=False):
        # --- 分支 1: Stage 1/2 获取文本特征 (CoOp Prompts) ---
        # Processor 调用: model(label=target, get_text=True)
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
            return text_features

        # --- 分支 2: Stage 1 获取图像特征 ---
        # Processor 调用: model(img, get_image=True)
        x = data_dict['imgs']
        image_features = self.image_encoder(x)['pooler_output']
        image_features_proj = self.visual_projection(image_features)
        if get_image:
            return image_features_proj


        feat = self.bottleneck(image_features)
        feat_proj = self.bottleneck_proj(image_features_proj)

        content_features = self.content_adapter(feat_proj)
        artifact_features = self.artifact_adapter(feat)

        if self.training:
            cls_score = self.classifier(artifact_features)
            # cls_score_proj = self.classifier_proj(feat_proj)

            feat_caption = None
            if captions is not None:
                with torch.no_grad():
                    tokenized_captions = self.tokenizer(
                        captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
                    ).to(x.device)
                    # 使用原始 CLIP 提取纯净语义
                    feat_caption = self.clip_model.get_text_features(**tokenized_captions)[1]

            return {
                "cls_score": cls_score,
                "feat": artifact_features,
                "image_features": artifact_features,
                "decouple_features":{
                    "content": content_features,
                    "artifact": artifact_features,
                    "caption": feat_caption
                }
            }
        else:
            if self.config["test"]["neck_feat"] == 'after':
                return torch.cat([artifact_features], dim=1)
            else:
                return torch.cat([image_features, image_features_proj], dim=1)

    def make_loss(self, config, num_classes=10):
        sampler = config['dataset']['sampler']

        # 1. 初始化基础损失函数
        if 'triplet' in config['model']['metric_loss_type']:
            if config['model']['no_margin']:
                triplet = TripletLoss()
            else:
                triplet = TripletLoss(config['solver']['margin'])
        else:
            triplet = None

        if config['model']['if_label_smooth'] == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            xent = nn.CrossEntropyLoss()

        # 2. 定义具体的 Loss 计算逻辑
        def loss_func(output_dict, target, i2tscore=None):
            """
            output_dict: forward 返回的字典
            target: 真实标签
            """
            # === A. 基础分类损失 (ID Loss) ===
            # cls_score 是 artifact 分支的预测结果
            score = output_dict['cls_score']

            # 兼容 score 是 list 的情况 (cls_score, cls_score_proj)
            if isinstance(score, list):
                ID_LOSS = [xent(scor, target) for scor in score[0:]]
                ID_LOSS = sum(ID_LOSS)
            else:
                ID_LOSS = xent(score, target)

            # === B. Triplet Loss (Metric Loss) ===
            # feat 是 artifact 分支的特征
            feat = output_dict['feat']
            if triplet is not None:
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = triplet(feat, target)[0]
            else:
                TRI_LOSS = 0.0

            if i2tscore != None:
                I2TLOSS = xent(i2tscore, target)
            else:
                I2TLOSS = 0.0

            # 计算总基础损失
            total_loss = config['model']['id_loss_weight'] * ID_LOSS + \
                         config['model']['triplet_loss_weight'] * TRI_LOSS + \
                         config['model']['i2t_loss_weight'] * I2TLOSS

            # === C. 解耦专用损失 (Decoupling Losses) ===
            # 只有在 output_dict 包含 'decouple_features' 时才计算 (即 Training 模式)
            if "decouple_features" in output_dict:
                decouple_dict = output_dict["decouple_features"]

                f_content = decouple_dict["content"]  # 归一化后的内容特征
                f_artifact = decouple_dict["artifact"]  # 归一化后的伪影特征
                f_caption = decouple_dict["caption"]  # 归一化后的 Caption 特征 (GT)

                # C.1 内容对齐损失 (Content Alignment Loss)
                # 目标: f_content 应该与 f_caption 越像越好
                # Loss = 1 - CosineSimilarity (因为输入已归一化，Cosine 就是点积)
                if f_caption is not None:
                    # dim=1 求和得到每个样本的点积，然后 mean
                    sim_content = torch.sum(f_content * f_caption, dim=1)
                    loss_content = 1.0 - torch.mean(sim_content)
                else:
                    loss_content = 0.0

                # C.2 正交解耦损失 (Orthogonality Loss)
                # 目标: f_content 和 f_artifact 应该正交 (不相关)
                # Loss = |CosineSimilarity| (即点积的绝对值越小越好)
                # 使用 .detach() 使得 f_content 不受此 loss 影响 (可选，通常为了稳定只更新 artifact adapter)
                # 这里我们让两者都更新，互相推开
                sim_orth = torch.sum(f_content * f_artifact, dim=1)
                loss_orth = torch.mean(torch.abs(sim_orth))

                # === D. 最终加权求和 ===
                # 从 config 中读取权重，如果没有则使用默认值
                w_content = config['model'].get('content_loss_weight', 0.1)
                w_orth = config['model'].get('orth_loss_weight', 0.1)

                total_loss = total_loss + 0 * loss_content + 0 * loss_orth

                # (可选) 打印调试信息，防止 Loss 爆炸
                # print(f"ID: {ID_LOSS.item():.4f}, Tri: {TRI_LOSS.item():.4f}, Cont: {loss_content.item():.4f}, Orth: {loss_orth.item():.4f}")

            return total_loss

        return loss_func

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
                                                                                 momentum=config['solver']['stage1'][
                                                                                     'momentum'])
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
                                                                                 momentum=config['solver']['stage2'][
                                                                                     'momentum'])
        elif config['solver']['stage2']['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=config['solver']['stage2']['base_lr'],
                                          weight_decay=config['solver']['stage2']['weight_decay'])
        else:
            optimizer = getattr(torch.optim, config['solver']['stage2']['type'])(params)
        return optimizer

    def save(self, save_path):
        if not osp.exists(save_path):
            os.makedirs(save_path)

            # 1. 保存 LoRA 权重 (PEFT 格式: adapter_config.json + adapter_model.bin)
            # 只有几 MB，非常轻量
        if hasattr(self.image_encoder, "save_pretrained"):
            self.image_encoder.save_pretrained(save_path)
            logger.info(f"Saved LoRA weights to {save_path}")

            # 2. 保存自定义组件 (Prompt Learner, Classifiers, BN)
        custom_dict = {
            "prompt_learner": self.prompt_learner.state_dict(),
            "classifier": self.classifier.state_dict(),
            "classifier_proj": self.classifier_proj.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "bottleneck_proj": self.bottleneck_proj.state_dict(),
        }

        custom_file = osp.join(save_path, "custom_components.pt")
        torch.save(custom_dict, custom_file)
        logger.info(f"Saved custom components to {custom_file}")

    def load(self, load_path):
        """
        load_path: 指向包含 adapter_config.json 的目录
        """
        # 1. 加载 LoRA 权重
        # 此时 self.image_encoder 已经是初始化好的 PeftModel (随机权重)
        # load_adapter 会覆盖掉随机权重
        if isinstance(self.image_encoder, PeftModel):
            logger.info(f"Loading LoRA weights from {load_path}")
            # adapter_name="default" 是 PEFT 的默认名称
            self.image_encoder.load_adapter(load_path, adapter_name="default")
        else:
            logger.warning("Current image_encoder is not a PeftModel, skipping LoRA load.")

        # 2. 加载自定义组件
        custom_file = osp.join(load_path, "custom_components.pt")
        if osp.exists(custom_file):
            logger.info(f"Loading custom components from {custom_file}")
            checkpoint = torch.load(custom_file, map_location="cpu")

            self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
            self.classifier.load_state_dict(checkpoint["classifier"])
            self.classifier_proj.load_state_dict(checkpoint["classifier_proj"])

            if "bottleneck" in checkpoint:
                self.bottleneck.load_state_dict(checkpoint["bottleneck"])
                self.bottleneck_proj.load_state_dict(checkpoint["bottleneck_proj"])
        else:
            logger.warning(f"Custom components file not found at {custom_file}")

class PromptLearner(nn.Module):
    def __init__(self, num_classes, class_names=None, clip_model=None):
        super().__init__()

        n_ctx = 16  # 学习 16 个 Context Token
        prompt_prefix = " ".join(["X"] * n_ctx) + " CLASS"

        token_embedding = clip_model.text_model.embeddings.token_embedding
        ctx_dim = clip_model.config.text_config.hidden_size

        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names
        self.num_classes = num_classes

        logger.info(f"Initializing Prompt Learner with {n_ctx} context vectors.")
        ctx_vectors = torch.empty(num_classes, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.cls_ctx = nn.Parameter(ctx_vectors)  # shape: [n_ctx, dim]

        logger.info(f"Number of context words (tokens): {n_ctx}")
        prompts = []
        for name in class_names:
            name = name.replace("_", " ")
            prompts.append(prompt_prefix + " " + name + ".")

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        tokenized_prompts = tokenizer(prompts, padding="max_length", max_length=77, truncation=True,
                                      return_tensors="pt").input_ids  # [n_cls, 77]
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:1, :1, :])

        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, label):
        """
        Args:
            label (torch.Tensor): 类别索引 [Batch]
        Returns:
            prompts (torch.Tensor): [Batch, 77, dim] - 这里的 prompts 是 input_embeds
            tokenized_prompts (torch.Tensor): [Batch, 77] - 对应的 input_ids (用于找EOT)
        """
        # 1. 获取 Learnable Context (Batch化)
        # label shape: [batch_size]
        # cls_ctx shape: [batch_size, n_ctx, dim]
        cls_ctx = self.cls_ctx[label]

        # 2. 获取 Prefix (SOS)
        # 简单起见，我们取第一个类别的 prefix 然后 expand (因为所有 SOS 都一样)
        # 或者如果你想支持 conditional prefix，可以用 self.token_prefix[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)  # [Batch, 1, dim]

        # 3. 获取 Suffix (Class Name + EOS) - 必须根据 Label 索引
        suffix = self.token_suffix[label]  # [Batch, *, dim]

        # 4. 拼接成完整的 Prompts Embeddings
        prompts = torch.cat(
            [
                prefix,  # (B, 1, dim)
                cls_ctx,  # (B, n_ctx, dim)
                suffix,  # (B, *, dim)
            ],
            dim=1,
        )
        return prompts


class TextEncoder(nn.Module):
    # ... (保持原有的 TextEncoder 代码) ...
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        attention_mask = (tokenized_prompts != 49407).long()
        bsz, seq_len = tokenized_prompts.shape
        extended_attention_mask = self.clip_model.get_extended_attention_mask(
            attention_mask,
            (bsz, seq_len),
            dtype=self.clip_model.dtype
        )
        hidden_states = self.text_model.embeddings(inputs_embeds=prompts.to(self.clip_model.device))
        encoder_outputs = self.text_model.encoder(hidden_states, attention_mask=extended_attention_mask)
        last_hidden_state = self.text_model.final_layer_norm(encoder_outputs.last_hidden_state)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        pooled_output = last_hidden_state[torch.arange(bsz), eot_indices]
        return self.text_projection(pooled_output)

    def encode_caption(self, text):
        """
        处理真实的文本 ID (通常用于 Validation 或 Zero-shot)
        text: [batch_size, 77] (tokenized indices)
        """
        # 直接使用 HF 的完整流程
        outputs = self.text_model(input_ids=text)
        x = outputs.last_hidden_state  # [batch_size, 77, d_model]

        # 取 EOT 特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        # 投影
        x = self.text_projection(x)

        return x


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        # 降维 -> 激活 -> 升维 (Bottleneck结构)
        hidden_dim = output_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        return self.net(x)+x