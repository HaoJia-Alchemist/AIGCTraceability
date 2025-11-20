#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HiDream模型生成器实现
"""

import logging
from typing import List, Any, Dict
import gc

import torch
from diffusers import HiDreamImagePipeline
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

from .base_generator import BaseGenerator

# 创建日志记录器
logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("HiDream_Generator")
class HiDreamGenerator(BaseGenerator):
    """
    HiDream图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化HiDream生成器

        Args:
            model_name: 模型名称
            model_params: 模型参数
        """
        super().__init__(model_name, model_params)
        self.pipeline = None
        self.default_params = {
            "model_type": "full",
            "height": 1024,
            "width": 1024,
            "guidance_scale": None,  # 如果为None，则使用模型配置中的默认值
            "num_inference_steps": None,  # 如果为None，则使用模型配置中的默认值
        }

    def load_model(self, device=None):
        """
        加载HiDream模型到指定设备

        Args:
            device: 设备
        """
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading model {self.model_name} on device {device}")
            dtype = torch.float16 if "cuda" in str(device) else torch.float32
            logger.info(f"Using device: {device}, dtype: {dtype}")
            tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            text_encoder_4 = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                              output_hidden_states=True, output_attentions=True,
                                                              torch_dtype=torch.bfloat16, )
            self.pipeline = HiDreamImagePipeline.from_pretrained("HiDream-ai/HiDream-I1-Full",# "HiDream-ai/HiDream-I1-Dev" | "HiDream-ai/HiDream-I1-Fast"
                                                        tokenizer_4=tokenizer_4,
                                                        text_encoder_4=text_encoder_4,
                                                        torch_dtype=torch.bfloat16, )
            self.pipeline = self.pipeline.to(device)

            # self.pipeline.set_progress_bar_config(disable=True)

            logger.info(f"Successfully loaded model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model on device {device}: {str(e)}")
            raise

    def generate(self, prompts: List[str], device=None) -> List[Any]:
        """
        使用HiDream批量生成图像

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
            result = self.pipeline( prompts,
                                    height=1024,
                                    width=1024,
                                    guidance_scale=5.0, # 0.0 for Dev&Fast
                                    num_inference_steps=50, # 28 for Dev and 16 for Fast
                                    )

            images = result.images
            return images
        except Exception as e:
            logger.error(f"Failed to generate images with {self.model_name} on device {device}: {str(e)}")
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