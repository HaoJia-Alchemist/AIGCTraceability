#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-Image生成器实现
"""

import gc
import logging
from typing import List, Any, Dict

import torch
from diffusers import Lumina2Pipeline, LCMScheduler, DiffusionPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download

from .base_generator import BaseGenerator

# 创建日志记录器
logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("DMD_Generator")
class DMD_Generator(BaseGenerator):
    """
    DMD_Generator图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化DMD_Generator生成器

        Args:
            model_name: 模型名称
            model_params: 模型参数
        """
        super().__init__(model_name, model_params)
        self.pipeline = None

    def load_model(self, device=None):
        """
        加载DMD_Generator模型到指定设备

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
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            repo_name = "tianweiy/DMD2"
            ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
            # Load model.
            unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(device, torch.float16)
            unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location=device))
            self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16,
                                                     variant="fp16").to(device)
            self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)

            logger.info(f"Successfully loaded model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model on device {device}: {str(e)}")
            raise e

    def generate(self, prompts: List[str], device=None) -> List[Any]:
        """
        使用DMD_Generator批量生成图像

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
                prompt=prompts, num_inference_steps=20, timesteps=[999, 749, 499, 249]
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
