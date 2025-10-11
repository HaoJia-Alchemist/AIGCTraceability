#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-Image生成器实现
"""

import logging
from typing import List, Any, Dict
import gc

import torch
from diffusers import OmniGenPipeline
from .base_generator import BaseGenerator

# 创建日志记录器
logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("OmniGen_Generator")
class OmniGen_Generator(BaseGenerator):
    """
    FLUX_1_Generator图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化FLUX_1_Generator生成器

        Args:
            model_name: 模型名称
            model_params: 模型参数
        """
        super().__init__(model_name, model_params)
        self.pipeline = None

    def load_model(self, device=None):
        """
        加载QwenImage模型到指定设备

        Args:
            device: 设备
        """
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading model {self.model_name} on device {device}")
            dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
            logger.info(f"Using device: {device}, dtype: {dtype}")

            # 从配置文件加载transformer，忽略不支持的参数
            self.pipeline = OmniGenPipeline.from_pretrained(
                "Shitao/OmniGen-v1-diffusers",
            ).to("cuda")

            # 将pipeline移动到指定设备
            self.pipeline = self.pipeline.to(device)

            logger.info(f"Successfully loaded model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model on device {device}: {str(e)}")
            raise

    def generate(self, prompts: List[str], device=None) -> List[Any]:
        """
        使用QwenImage批量生成图像

        Args:
            prompts: 文本提示列表
            device: 使用的设备

        Returns:
            生成的图像列表
        """
        try:
            if self.pipeline is None:
                self.load_model(device)

            # 使用pipeline的批量生成功能
            result = self.pipeline(
                prompts,
                num_inference_steps=50,
                width=1024,
                height=1024
            )

            images = result.images
            return images
        except Exception as e:
            logger.error(f"Failed to generate images with QwenImage on device {device}: {str(e)}")
            raise

    def clear_model(self):
        """
        清理模型占用的资源
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()