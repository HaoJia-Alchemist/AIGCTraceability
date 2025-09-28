#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-Image生成器实现
"""

import logging
from typing import List, Any, Dict
import gc

import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
from .base_generator import BaseGenerator

# 创建日志记录器
logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("QwenImage_Generator")
class QwenImage_Generator(BaseGenerator):
    """
    QwenImage_Generator图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化QwenImage_Generator生成器

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

            ckpt_path = self.model_params.get("gguf_quantization_file")
            
            # 从配置文件加载transformer，忽略不支持的参数
            transformer = QwenImageTransformer2DModel.from_single_file(
                ckpt_path,
                config="Qwen/Qwen-Image",
                subfolder="transformer",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            
            # 创建pipeline
            self.pipeline = QwenImagePipeline.from_pretrained(
                self.model_name,
                transformer=transformer,
                torch_dtype=dtype,
            )

            # 将pipeline移动到指定设备
            self.pipeline = self.pipeline.to(device)

            # self.pipeline.set_progress_bar_config(disable=True)

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
                height=self.model_params.get('height', 1024),
                width=self.model_params.get('width', 1024),
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