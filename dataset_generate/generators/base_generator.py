#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成器基类
"""

import abc
from typing import List, Dict, Any
import torch


class BaseGenerator(metaclass=abc.ABCMeta):
    """
    AIGC图像生成器基类
    """
    
    # 生成器注册表
    _generators = {}

    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            model_params: 模型参数
        """
        self.model_name = model_name
        self.model_params = model_params

    @abc.abstractmethod
    def generate(self, prompts: List[str], device=None) -> List[Any]:
        """
        根据文本提示生成图像
        
        Args:
            prompts: 文本提示列表
            device: 使用的设备
            
        Returns:
            生成的图像列表
        """
        pass

    @abc.abstractmethod
    def load_model(self, device=None):
        """
        加载模型到指定设备
        
        Args:
            device: 设备
        """
        pass

    def clear_model(self):
        """
        清理模型占用的资源
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        torch.cuda.empty_cache()

    @classmethod
    def register(cls, name: str):
        """
        注册生成器类的装饰器
        
        Args:
            name: 生成器名称
        """
        def decorator(generator_cls):
            cls._generators[name] = generator_cls
            return generator_cls
        return decorator

    @classmethod
    def create(cls, name: str, model_name: str, model_params: Dict[str, Any]):
        """
        根据名称创建生成器实例
        
        Args:
            name: 生成器名称
            model_name: 模型名称
            model_params: 模型参数
            
        Returns:
            生成器实例
        """
        if name not in cls._generators:
            raise ValueError(f"Unknown generator: {name}")
        
        generator_cls = cls._generators[name]
        return generator_cls(model_name, model_params)