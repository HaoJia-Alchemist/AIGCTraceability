import math
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

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


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@MODEL_FACTORY.register_module(module_name='Prompt_Caption_Decoupling')
class Prompt_Caption_Decoupling(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(Prompt_Caption_Decoupling, self).__init__(config, num_classes)
        model_name = "openai/clip-vit-large-patch14"
        self.in_planes = 1024
        self.in_planes_proj = 768

        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_model.vision_model = apply_svd_residual_to_self_attn(self.clip_model.vision_model, r=1024 - 1)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

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

        self.classifier_artiface = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_artiface.apply(weights_init_classifier)

        self.bottleneck_artiface = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_artiface.bias.requires_grad_(False)
        self.bottleneck_artiface.apply(weights_init_kaiming)

        self.clip_model.vision_model = apply_svd_residual_to_self_attn(self.clip_model.vision_model, r=1024 - 1)
        self.image_encoder = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection

        self.prompt_learner = PromptLearner(num_classes, clip_model=self.clip_model)
        self.text_encoder = TextEncoder(self.clip_model)

        self.content_adapter = ResidualAdapter(self.in_planes_proj, self.in_planes_proj)
        # self.artifact_adapter = ResidualAdapter(self.in_planes_proj, self.in_planes_proj)
        self.artifact_adapter = AttentiveArtifactExtractor(self.in_planes, self.in_planes_proj)

        self.content_adapter.apply(weights_init_kaiming)
        self.artifact_adapter.apply(weights_init_kaiming)


    def forward(self, data_dict=None, label=None, captions=None, get_image=False, get_text=False):
        # --- 分支 1: Stage 1/2 获取文本特征 (CoOp Prompts) ---
        # Processor 调用: model(label=target, get_text=True)
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
            return text_features

        x = data_dict['imgs']
        # --- 分支 2: Stage 1 获取图像特征 ---
        # Processor 调用: model(img, get_image=True)
        if get_image:
            image_features = self.image_encoder(x)['pooler_output']
            image_features_proj = self.visual_projection(image_features)
            return image_features_proj

        vision_outputs = self.image_encoder(x)
        last_hidden_state = vision_outputs.last_hidden_state

        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]
        cls_features = self.clip_model.vision_model.post_layernorm(cls_token)
        cls_features_proj = self.visual_projection(cls_features)

        content_features = self.content_adapter(cls_features_proj)
        artifact_features = self.artifact_adapter(patch_tokens)

        feat = self.bottleneck(cls_features)
        feat_proj = self.bottleneck_proj(cls_features_proj)
        feat_artiface = self.bottleneck_artiface(artifact_features)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            cls_score_artiface = self.classifier_artiface(feat_artiface)

            if captions is not None:
                with torch.no_grad():
                    tokenized_captions = self.tokenizer(
                        captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
                    ).to(x.device)
                    # 使用原始 CLIP 提取纯净语义
                    captions_features = self.clip_model.get_text_features(**tokenized_captions)[1]

            return {
                "cls_score": [ cls_score, cls_score_proj],#, cls_score_artiface],
                "image_features": [cls_features, cls_features_proj],#, artifact_features],
                "image_features_proj": cls_features_proj,
                # "decouple_features":{
                #     "content": content_features,
                #     "artifact": artifact_features,
                #     "caption": captions_features
                # }
            }
        else:
            if self.config["test"]["neck_feat"] == 'after':
                return feat
            else:
                return image_features
            if self.config["test"]["neck_feat"] == 'after':
                return torch.cat([feat, feat_proj], dim=1)
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
        def loss_func(output_dict, data_dict, i2tscore=None):
            cls_score = output_dict['cls_score']
            image_features = output_dict['image_features']
            target = data_dict['df_ids']
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

            # if i2tscore != None:
            #     I2TLOSS = xent(i2tscore, target)
            #     loss = config['model']['i2t_loss_weight'] * I2TLOSS + loss


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
                    sim_content = torch.nn.functional.cosine_similarity(f_content,f_caption)
                    loss_content = 1 - sim_content.mean()
                else:
                    loss_content = 0.0

                f_content_norm = F.normalize(f_content, p=2, dim=1)
                f_artifact_norm = F.normalize(f_artifact, p=2, dim=1)
                dot_product = (f_content_norm * f_artifact_norm).sum(dim=1)
                loss_orthogonal = (dot_product ** 2).mean()


                # === D. 最终加权求和 ===
                # 从 config 中读取权重，如果没有则使用默认值
                w_content = config['model'].get('content_loss_weight', 0.1)
                w_orth = config['model'].get('orth_loss_weight', 0.1)

                loss = loss + w_content * loss_content + w_orth * loss_orthogonal

                # (可选) 打印调试信息，防止 Loss 爆炸
                # print(f"ID: {ID_LOSS.item():.4f}, Tri: {TRI_LOSS.item():.4f}, Cont: {loss_content.item():.4f}, Orth: {loss_orth.item():.4f}")

            return loss

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

# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)


# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]  # Shape: (out_features, r)
        S_r = S[:r]  # Shape: (r,)
        Vh_r = Vh[:r, :]  # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r

        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]  # Shape: (out_features, n - r)
        S_residual = S[r:]  # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module


class PromptLearner(nn.Module):
    def __init__(self, num_classes, class_names=None, clip_model=None):
        super().__init__()

        n_ctx = 4  # 学习 16 个 Context Token
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

        self.register_buffer("token_prefix", embedding[:1, :1, :].contiguous())

        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :].contiguous())

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
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.hidden_size = clip_model.config.text_config.hidden_size

    def forward(self, prompts, tokenized_prompts):
        """
        prompts: [batch_size, n_ctx, d_model] - 这里的 prompts 通常是类似 CoOp 的 learnable vectors
        tokenized_prompts: [batch_size, n_ctx] - token id，用于定位 EOT
        """
        attention_mask = (tokenized_prompts != 49407).long()
        hidden_states = self.text_model.embeddings(inputs_embeds=prompts)
        bsz, seq_len = tokenized_prompts.shape
        extended_attention_mask = self.clip_model.get_extended_attention_mask(
            attention_mask,
            (bsz, seq_len),
            dtype=self.clip_model.dtype
        )
        encoder_outputs = self.text_model.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        pooled_output = last_hidden_state[torch.arange(bsz), eot_indices]
        text_features = self.text_projection(pooled_output)
        return text_features

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
