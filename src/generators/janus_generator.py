#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Janus模型生成器实现
"""

import logging
import os
from typing import List, Any, Dict
import gc

import torch
import PIL.Image
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

from .base_generator import BaseGenerator

# 创建日志记录器
logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("Janus_Generator")
class JanusGenerator(BaseGenerator):
    """
    Janus模型图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化Janus生成器

        Args:
            model_name: 模型名称
            model_params: 模型参数
        """
        super().__init__(model_name, model_params)
        self.vl_chat_processor = None
        self.model = None
        self.default_params = {
            "temperature": 1.0,
            "parallel_size": 1,
            "cfg_weight": 5.0,
            "image_token_num_per_image": 576,
            "img_size": 384,
            "patch_size": 16,
        }

    def load_model(self, device=None):
        """
        加载Janus模型到指定设备

        Args:
            device: 设备
        """
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading Janus model {self.model_name} on device {device}")
            
            # 加载处理器
            self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_name)
            
            # 加载模型
            self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 转换模型精度并移到指定设备
            self.model = self.model.to(torch.bfloat16).to(device).eval()
            
            logger.info(f"Successfully loaded Janus model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load Janus model on device {device}: {str(e)}")
            raise

    def generate(self, prompts: List[str], device=None) -> List[Any]:
        """
        使用Janus模型批量生成图像

        Args:
            prompts: 文本提示列表
            device: 使用的设备

        Returns:
            生成的图像列表
        """
        try:
            if self.model is None or self.vl_chat_processor is None:
                self.load_model(device)
            
            # 获取生成参数
            gen_params = {**self.default_params, **self.model_params}
            
            images = []
            for prompt in prompts:
                # 构建对话
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": prompt,
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]

                # 应用模板
                sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
                prompt_text = sft_format + self.vl_chat_processor.image_start_tag
                
                # 生成图像
                generated_images = self._generate_single(
                    prompt_text, 
                    **gen_params
                )
                
                images.extend(generated_images)
                
            return images
        except Exception as e:
            logger.error(f"Failed to generate images with Janus model on device {device}: {str(e)}")
            raise

    @torch.inference_mode()
    def _generate_single(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        image_token_num_per_image: int = 4096,
        img_size: int = 1024,
        patch_size: int = 16,
    ):
        """
        生成单个图像
        
        Args:
            prompt: 提示文本
            temperature: 温度参数
            parallel_size: 并行生成数量
            cfg_weight: 分类器自由引导权重
            image_token_num_per_image: 每张图像的token数量
            img_size: 图像尺寸
            patch_size: patch尺寸
            
        Returns:
            生成的图像列表
        """
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        pil_images = []
        for i in range(parallel_size):
            pil_image = PIL.Image.fromarray(visual_img[i])
            pil_images.append(pil_image)

        return pil_images

    def clear_model(self):
        """
        清理模型占用的资源
        """
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.vl_chat_processor is not None:
            del self.vl_chat_processor
            self.vl_chat_processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()