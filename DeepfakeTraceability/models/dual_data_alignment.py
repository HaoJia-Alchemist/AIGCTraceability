import torch
import torch.nn as nn
from . import MODEL_FACTORY

from .base_model import BaseModel


@MODEL_FACTORY.register_module(module_name='dual_data_alignment')
class Dual_Data_Alignment(BaseModel):
    def __init__(self, config=None, num_classes=10):
        super(Dual_Data_Alignment, self).__init__(config, num_classes)
        self.config = config
        self.epoch = 0
        self.num_classes = num_classes
        self.backbone = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1, lora_targets=None)

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint['model'])


    def forward(self, data_dict) -> dict:
        x = data_dict.get('imgs')
        img_feature,fc = self.backbone(x, return_feature=True)

        if self.training:
            cls_score = self.classifier(feat)
            return {"cls_score": cls_score,
                    "image_features": feat}

        else:
            return img_feature


import torch.nn as nn



class DINOv2ModelWithLoRA(nn.Module):
    def __init__(self, name, num_classes=1, lora_rank=8, lora_alpha=1.0, lora_targets=None):
        super(DINOv2ModelWithLoRA, self).__init__()

        # Create the base model with all parameters including the new ones
        self.base_model = DINOv2Model(
            name=name,
            num_classes=num_classes,
        )

        self.name = name


        if lora_targets is None:
            lora_targets = ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']

        # apply LoRA
        print(f"Adding LoRA to DINOv2 (rank={lora_rank}, alpha={lora_alpha})")
        print(f"LoRA target modules: {lora_targets}")
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
            trainable_orig=False
        )

        self._get_lora_params = lambda: get_lora_params(self.base_model.model)

    def get_trainable_params(self):
        """
        Get trainable parameters for optimization.

        Returns:
            list: List of parameters that should be updated during training
        """
        lora_params = self._get_lora_params()
        fc_params = self.base_model.fc.parameters()
        total_lora_params = sum(p.numel() for p in lora_params)
        total_fc_params = sum(p.numel() for p in fc_params)
        return list(lora_params) + list(fc_params)

    def forward(self, x, return_feature=False, return_tokens=False):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor
            return_feature (bool): Whether to return features
            return_tokens (bool): Whether to return token features

        Returns:
            Various outputs depending on return_feature and return_tokens flags
        """
        return self.base_model(x, return_feature=return_feature, return_tokens=return_tokens)

    def get_preprocessing_transforms(self):
        """
        Get the preprocessing transforms for the model.

        Returns:
            torchvision.transforms.Compose: Preprocessing transforms
        """
        return self.base_model.get_preprocessing_transforms()


import torch
import torch.nn as nn
import torch.hub

CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Model(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOv2Model, self).__init__()
        print(f"Loading DINOv2 from hub: {name}")
        self.model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False, return_tokens=False):
        if hasattr(self.model, 'forward_features'):
            features_dict = self.model.forward_features(x)
            features = features_dict['x_norm_clstoken']
        else:
            features = self.model(x)
            if isinstance(features, dict):
                features = features.get('x_norm_clstoken', features.get('last_hidden_state', None)[:, 0])

        if return_feature:
            return features, self.fc(features)

        return self.fc(features)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank

        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_out = torch.einsum('...d, rd -> ...r', x, self.lora_A)  # x @ A.T
        lora_out = torch.einsum('...r, or -> ...o', lora_out, self.lora_B)  # (x @ A.T) @ B.T

        return lora_out * (self.alpha / self.rank)


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_orig=False):
        super().__init__()
        self.original_layer = original_layer

        if not trainable_orig:
            for param in self.original_layer.parameters():
                param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora(x)

        if original_output.shape != lora_output.shape:
            raise ValueError(
                f"dimension mismatch: original output {original_output.shape}, LoRA output {lora_output.shape}")

        return original_output + lora_output

    def __getattr__(self, name):
        if name == 'weight':
            return self.original_layer.weight
        elif name == 'bias':
            return self.original_layer.bias
        else:
            return super().__getattr__(name)


def apply_lora_to_linear_layers(model, rank=4, alpha=1.0, target_modules=None, trainable_orig=False):
    for name, module in model.named_modules():
        if target_modules is not None and not any(target in name for target in target_modules):
            continue

        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            parent = model if parent_name == '' else get_submodule(model, parent_name)

            setattr(parent, child_name, LoRALinear(module, rank, alpha, trainable_orig))

    return model


def get_submodule(model, submodule_name):
    """Helper function to get a submodule from a model"""
    if not submodule_name:
        return model

    parts = submodule_name.split('.')
    current_module = model

    for part in parts:
        if part.isdigit():
            current_module = current_module[int(part)]
        else:
            current_module = getattr(current_module, part)

    return current_module


def get_lora_params(model):
    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
        elif isinstance(module, LoRALinear):
            lora_params.extend(module.lora.parameters())

    return lora_params