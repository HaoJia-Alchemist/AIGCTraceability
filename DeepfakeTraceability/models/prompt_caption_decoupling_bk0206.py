import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from peft import LoraConfig, get_peft_model
import transformers

from loss import TripletLoss, CrossEntropyLabelSmooth
from .effort_model import apply_svd_residual_to_self_attn

# 屏蔽烦人的 Warning
transformers.logging.set_verbosity_error()

from . import MODEL_FACTORY
from .base_model import BaseModel

logger = logging.getLogger(__name__)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
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
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ==========================================
# 改进模块 1: 增强型 Adapter (Residual MLP)
# ==========================================
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
        self.skip_connection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        return self.net(x)+self.skip_connection(x)


# ==========================================
# 改进模块 2: 伪影特征提取器 (Patch Statistics)
# ==========================================
class AttentiveArtifactExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 1. 注意力模块: 学习每个 Patch 的重要性权重
        self.attn_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

        # 2. 特征映射 (输入依然是 input_dim, 因为我们做的是加权和，而不是拼接)
        # 如果你依然想保留 std 特征，可以拼接 Weighted_Mean + Std
        self.adapter = ResidualAdapter(input_dim * 2, output_dim)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, N, D]
        """
        # --- A. 计算空间注意力 ---
        # attn_scores: [B, N, 1]
        attn_scores = self.attn_proj(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # --- B. 加权聚合 (Weighted Pooling) ---
        # [B, N, 1] * [B, N, D] -> sum(dim=1) -> [B, D]
        weighted_feat = (patch_tokens * attn_weights).sum(dim=1)

        # --- C. 补充统计特征 (Std Pooling) ---
        # 标准差依然很有用，能捕捉噪声强度分布
        std_feat = patch_tokens.std(dim=1)

        # --- D. 拼接 & 映射 ---
        combined = torch.cat([weighted_feat, std_feat], dim=1)  # [B, 2048]
        return self.adapter(combined)


@MODEL_FACTORY.register_module(module_name='Prompt_Caption_Decoupling')
class Prompt_Caption_Decoupling(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(Prompt_Caption_Decoupling, self).__init__(config, num_classes)
        model_name = "openai/clip-vit-large-patch14"
        self.vision_width = 1024
        self.embed_dim = 768  # CLIP projection dim

        # 1. 加载模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # self.clip_model.vision_model = apply_svd_residual_to_self_attn(self.clip_model.vision_model, r=1024 - 1)

        # 2. 视觉编码器 + LoRA (使用 PEFT 替代手写 SVD)
        # ----------------------------------------------------
        self.image_encoder = self.clip_model.vision_model


        # peft_config = LoraConfig(
        #     r=8,  # Rank 增大一点以提升拟合能力
        #     lora_alpha=8,
        #     target_modules=["q_proj", "v_proj"],#, "out_proj", "fc1", "fc2"],  # 同时微调 Attn 和 MLP
        #     lora_dropout=0.1,
        #     bias="none"
        # )
        # self.image_encoder = get_peft_model(self.image_encoder, peft_config)
        # self.image_encoder.print_trainable_parameters()

        self.visual_projection = self.clip_model.visual_projection

        # 3. 解耦适配器 (优化)
        # ----------------------------------------------------
        # Content Branch: 使用投影后的语义特征 (768)
        self.content_adapter = ResidualAdapter(self.embed_dim, self.embed_dim)

        # Artifact Branch: 使用原始 Patch 特征 (1024 -> 2048 -> 768)
        # self.artifact_adapter = AttentiveArtifactExtractor(self.vision_width, self.embed_dim)
        self.artifact_adapter = ResidualAdapter(1024, 1024)
        # 4. 分类器和 BN
        self.classifier_proj = nn.Linear(1024, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck_proj = nn.BatchNorm1d(1024)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # 5. Prompt Learner & Text Encoder
        self.prompt_learner = PromptLearner(num_classes, clip_model=self.clip_model, tokenizer=self.tokenizer)
        self.text_encoder = TextEncoder(self.clip_model)

        # 确保初始化
        self.content_adapter.apply(weights_init_classifier)
        self.artifact_adapter.apply(weights_init_classifier)

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

    def forward(self, data_dict=None, label=None, captions=None, get_image=False, get_text=False):
        # [Branch 1] Text Features (Stage 1/2)
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
            return text_features

        x = data_dict['imgs']
        # [Branch 2] Image Features
        # x: [Batch, 3, 224, 224]
        # 使用 output_hidden_states=True 或者直接取 last_hidden_state
        vision_outputs = self.image_encoder(x)

        # last_hidden_state: [Batch, 257, 1024] (1 CLS + 256 Patches)
        last_hidden_state = vision_outputs.last_hidden_state

        # A. 提取全局语义 (CLS Token) -> 用于 Content
        cls_token = last_hidden_state[:, 0, :]  # [Batch, 1024]
        image_features = self.clip_model.vision_model.post_layernorm(cls_token)
        # 经过 CLIP 原始投影 (1024 -> 768)
        image_features_proj = self.visual_projection(image_features)

        if get_image:
            # 推理时，可以直接返回 artifact 特征，或者融合特征
            # 这里为了简单，先按原始逻辑返回 semantic
            return image_features_proj

        # B. 提取局部伪影 (Patch Tokens) -> 用于 Artifact
        # patch_tokens = last_hidden_state[:, 1:, :]  # [Batch, 256, 1024]

        # --- 解耦过程 ---

        # 1. Content: 基于语义特征微调
        feat_content = self.content_adapter(image_features_proj)

        # 2. Artifact: 基于 Patch 统计特征提取
        # 相比只用 CLS token，Patch token 保留了更多生成痕迹（如棋盘格效应、边缘噪声）
        feat_artifact = self.artifact_adapter(image_features)

        # 归一化
        feat_content_norm = feat_content / (feat_content.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        feat_artifact_norm = feat_artifact / (feat_artifact.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        # 溯源分类: 只使用 Artifact 特征
        feat_artifact_bn = self.bottleneck_proj(feat_artifact)

        if self.training:
            cls_score_proj = self.classifier_proj(feat_artifact_bn)

            # Caption 特征 (Ground Truth Content)
            feat_caption = None
            if captions is not None:
                with torch.no_grad():
                    tokenized_captions = self.tokenizer(
                        captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
                    ).to(x.device)
                    # 使用原始 CLIP 提取纯净语义
                    feat_caption = self.clip_model.get_text_features(**tokenized_captions)[1]
                    feat_caption = feat_caption / (feat_caption.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            return {
                "cls_score": cls_score_proj,
                "feat": feat_artifact,  # 用于 Triplet Loss
                "image_features": image_features,
                "decouple_features": {
                    "content": feat_content_norm,
                    "artifact": feat_artifact_norm,
                    "caption": feat_caption
                }
            }
        else:
            # 测试阶段
            if self.config["test"]["neck_feat"] == 'after':
                return feat_artifact_bn
            else:
                return feat_artifact_norm

    # 优化器配置
    def make_optimizer_stage1(self, config, model):
        # 保持不变，只训练 Prompt Learner
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
            # 1. 明确不需要训练的部分
            if "text_encoder" in key: continue
            if "prompt_learner" in key: continue

            # 2. 筛选需要训练的部分 (LoRA + Adapters + Classifier)
            if not value.requires_grad: continue

            lr = config['solver']['stage2']['base_lr']
            weight_decay = config['solver']['stage2']['weight_decay']

            if "bias" in key:
                lr *= config['solver']['stage2']['bias_lr_factor']
                weight_decay = config['solver']['stage2']['weight_decay_bias']

            # 分类器通常需要更大的 LR
            if config['solver']['stage2']['large_fc_lr']:
                if "classifier" in key or "adapter" in key:  # 让 adapter 也享受大 LR
                    lr *= 2

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

# 辅助类保持不变
class PromptLearner(nn.Module):
    # ... (保持原有的 PromptLearner 代码，记得加上 .contiguous() 修复) ...
    def __init__(self, num_classes, class_names=None, clip_model=None, tokenizer=None):
        super().__init__()

        n_cls_ctx = 8
        ctx_dim = clip_model.config.text_config.hidden_size

        logger.info(f"Initializing Prompt Learner with {n_cls_ctx} context vectors.")
        ctx_vectors = torch.empty(num_classes, n_cls_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(ctx_vectors)

        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names
        self.num_classes = num_classes
        n_ctx = 4
        prompt_prefix = "A photo generated by"
        prompts = []
        for name in class_names:
            name = name.replace("_", " ")
            prompts.append(prompt_prefix + " " + name + "AIGC model.")
        tokenized_prompts = tokenizer(prompts, padding="max_length", max_length=77, truncation=True,
                                      return_tensors="pt").input_ids  # [n_cls, 77]
        token_embedding = clip_model.text_model.embeddings.token_embedding
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:1, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.n_ctx = n_ctx
        self.n_cls_ctx = n_cls_ctx
        self.num_classes = num_classes

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
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
        ).to(device=label.device)

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