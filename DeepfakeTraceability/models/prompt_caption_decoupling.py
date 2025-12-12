import math
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

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




@MODEL_FACTORY.register_module(module_name='Prompt_Caption_Decoupling')
class Prompt_Caption_Decoupling(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(Prompt_Caption_Decoupling, self).__init__(config, num_classes)
        model_name = "openai/clip-vit-large-patch14"
        self.in_planes = 1024
        self.in_planes_proj = 768

        self.clip_model = CLIPModel.from_pretrained(model_name)
        text_processor = CLIPProcessor.from_pretrained(model_name)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.clip_model.vision_model = apply_svd_residual_to_self_attn(self.clip_model.vision_model, r=1024 - 1)
        self.image_encoder = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection

        # === Tokenizer ===
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.prompt_learner = PromptLearner(num_classes, clip_model=self.clip_model, tokenizer=self.tokenizer)
        self.text_encoder = TextEncoder(self.clip_model)

        # === 解耦映射头 ===
        self.content_adapter = FeatureDecoupler(self.in_planes_proj, self.in_planes_proj)
        self.artifact_adapter = FeatureDecoupler(self.in_planes_proj, self.in_planes_proj)
        # 初始化权重
        self.content_adapter.apply(weights_init_classifier)  # 使用分类器初始化方式即可
        self.artifact_adapter.apply(weights_init_classifier)

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
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)

        elif sampler == 'softmax_triplet':
            def loss_func(output_dict, target, i2tscore=None):
                score, feat = output_dict['score'], output_dict['feat']
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

                        loss = config['model']['id_loss_weight'] * ID_LOSS + config['model'][
                            'triplet_loss_weight'] * TRI_LOSS

                        if i2tscore != None:
                            I2TLOSS = xent(i2tscore, target)
                            loss = config['model']['i2t_loss_weight'] * I2TLOSS + loss

                        # === 解耦损失 ===
                        decouple_dict = output_dict["decouple_features"]
                        f_content = decouple_dict["content"]
                        f_artifact = decouple_dict["artifact"]
                        f_caption = decouple_dict["caption"]
                        content_loss = 1.0 - torch.mean(torch.sum(f_content * f_caption, dim=1))
                        orth_loss = torch.mean(torch.sum(f_content * f_artifact, dim=1) ** 2)
                        loss = loss + config['model']['content_loss'] * content_loss + config['model']['orth_loss'] * orth_loss
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


    def forward(self, x=None, label=None, captions=None, get_image=False, get_text=False):
        # --- 分支 1: Stage 1/2 获取文本特征 (CoOp Prompts) ---
        # Processor 调用: model(label=target, get_text=True)
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
            return text_features

        # --- 分支 2: Stage 1 获取图像特征 ---
        # Processor 调用: model(img, get_image=True)

        image_features = self.image_encoder(x)['pooler_output']
        image_features_proj = self.visual_projection(image_features)

        if get_image:
            return image_features_proj

        # 提取内容特征 (F_content)
        feat_content = self.content_adapter(image_features_proj)
        # 提取伪影特征 (F_artifact)
        feat_artifact = self.artifact_adapter(image_features_proj)

        feat_content_norm = feat_content / feat_content.norm(p=2, dim=-1, keepdim=True)
        feat_artifact_norm = feat_artifact / feat_artifact.norm(p=2, dim=-1, keepdim=True)

        feat_artifact_proj = self.bottleneck_proj(feat_artifact)
        if self.training:
            cls_score_proj = self.classifier_proj(feat_artifact_proj)
            with torch.no_grad():
                # 我们希望 Caption 特征是 Ground Truth，所以不计算梯度，或者根据需求开启梯度
                # 这里假设用原始 CLIP Text Encoder 提取标准语义
                tokenized_captions = self.tokenizer(
                    captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
                ).to(image_features.device)
                feat_caption = self.clip_model.get_text_features(**tokenized_captions)
                feat_caption = feat_caption / feat_caption.norm(p=2, dim=-1, keepdim=True)
            return {
                "cls_score": cls_score_proj,
                "feat": feat_artifact,
                "image_features": image_features_proj,
                "decouple_features":{
                    "content": feat_content_norm,
                    "artifact": feat_artifact_norm,
                    "caption": feat_caption
                }
            }
        else:
            if self.config["test"]["neck_feat"] == 'after':
                return feat_artifact_proj
            else:
                return feat_artifact

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

    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()

            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)

            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0

        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
        else:
            loss = 0.0

        return loss

    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = weight_current_fnorm ** 2
        else:
            loss = 0.0

        return loss


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
    def __init__(self, num_classes, class_names=None, clip_model=None, tokenizer=None):
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
            attention_mask, (bsz, seq_len), prompts.device
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

class FeatureDecoupler(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)