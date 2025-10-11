#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成器模块初始化文件
"""

# 导入所有生成器以确保它们被正确注册
from .base_generator import BaseGenerator
from .diffusion_pipline_generator import Diffusion_Pipline_Generator
from .qwen_image_generator import QwenImage_Generator
from .flux_1_generator import FLUX_1_Generator
from .sdxl_pipline_generator import StableDiffusionXL_Pipline_Generator
from .janus_generator import JanusGenerator
from .hidream_generator import HiDreamGenerator
from .kolors_generator import Kolors_Generator
from .lumina2_generator import Lumina2_Generator
from .dmd_generator import DMD_Generator
from .omnigen_generator import OmniGen_Generator
__all__ = [
    "BaseGenerator",
    "Diffusion_Pipline_Generator",
    "QwenImage_Generator",
    "FLUX_1_Generator",
    "StableDiffusionXL_Pipline_Generator",
    "JanusGenerator",
    "HiDreamGenerator",
    "Kolors_Generator",
    "Lumina2_Generator",
    "DMD_Generator",
    "OmniGen_Generator"
]