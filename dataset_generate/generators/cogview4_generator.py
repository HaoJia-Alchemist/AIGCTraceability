from typing import Dict, Any, List

import torch
from diffusers import CogView4Pipeline, DiffusionPipeline
from .base_generator import BaseGenerator
import logging
import gc

logger = logging.getLogger("ImageGeneration")


@BaseGenerator.register("CogView4_Pipline_Generator")
class CogView4_Pipline_Generator(BaseGenerator):
    """
    ProteusV06_Pipline_Generator图像生成器
    """

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化ProteusV06_Pipline_Generator生成器

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

            self.pipeline = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

            # Open it for reduce GPU memory usage
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()

            # Configure the pipeline
            # self.pipeline = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

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
