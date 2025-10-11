#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据生成脚本，根据配置文件调用不同的生成器生成text-to-image数据
"""

import os
import csv
import yaml
import argparse
import torch
from typing import List, Dict, Any
import time
import logging
from logging.handlers import RotatingFileHandler

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators import BaseGenerator

from accelerate import Accelerator
from accelerate.utils import gather_object
ACCELERATE_AVAILABLE = True


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        config: 日志配置
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_file = config.get('file', 'logs/generation.log')
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('ImageGeneration')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器（支持轮转）
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def load_prompts(file_path: str) -> List[Dict[str, str]]:
    """
    从prompt_combined.txt文件加载文本描述
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含图像文件名和文本描述的字典列表
    """
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'image': row['image'],
                'prompt': row['prompt']
            })
    return prompts


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def batch_prompts(prompts: List[Dict[str, str]], batch_size: int) -> List[Dict[str, Any]]:
    """
    将文本描述分批处理
    
    Args:
        prompts: 文本描述列表
        batch_size: 批次大小
        
    Returns:
        分批后的文本描述列表，每个批次包含数据和ID
    """
    batches = []
    for i in range(0, len(prompts), batch_size):
        batch = {
            'data': prompts[i:i + batch_size],
            'id': i // batch_size
        }
        batches.append(batch)
    return batches


def save_images(images: List[Any], batch: Dict[str, Any], output_dir: str, 
                generator_type: str, num_images_per_prompt: int = 1, logger: logging.Logger = None):
    """
    保存生成的图像
    
    Args:
        images: 生成的图像列表
        batch: 批次数据
        output_dir: 输出目录
        generator_type: 生成器类型
        num_images_per_prompt: 每个prompt生成的图像数量
        logger: 日志记录器
    """
    # 确保输出目录存在
    output_path = os.path.join(output_dir, generator_type)
    os.makedirs(output_path, exist_ok=True)
    
    prompts = batch['data']
    batch_id = batch['id']
    
    img_idx = 0
    saved_count = 0
    for prompt_data in prompts:
        image_name = prompt_data['image']
        
        # 当num_images_per_prompt为1时，保持文件名与prompt_combined.txt中的一致
        # 当num_images_per_prompt大于1时，才添加索引后缀
        if num_images_per_prompt <= 1:
            # 保持文件名与原始文件名一致
            save_path = os.path.join(output_path, image_name)
            if img_idx < len(images):
                # 实际使用时取消注释下面这行
                images[img_idx].save(save_path)
                message = f"Saving image to {save_path}"
                if logger:
                    logger.info(message)
                else:
                    print(message)
                img_idx += 1
                saved_count += 1
        else:
            # 为每张图片添加索引后缀
            name, ext = os.path.splitext(image_name)
            for i in range(num_images_per_prompt):
                if img_idx < len(images):
                    final_name = f"{name}_{i}{ext}"
                    save_path = os.path.join(output_path, final_name)
                    # 实际使用时取消注释下面这行
                    images[img_idx].save(save_path)
                    message = f"Saving image to {save_path}"
                    if logger:
                        logger.info(message)
                    else:
                        print(message)
                    img_idx += 1
                    saved_count += 1
    
    message = f"Batch {batch_id}: Saved {saved_count} images to {output_path}"
    if logger:
        logger.info(message)
    else:
        print(message)


def process_generator_single_gpu(generator_type: str, generator_config: Dict[str, Any],
                                 batches: List[Dict[str, Any]], data_config: Dict[str, Any],
                                 output_dir: str, logger: logging.Logger):
    """
    单GPU处理生成器的图像生成（当accelerate不可用时的回退方案）
    
    Args:
        generator_type: 生成器类型
        generator_config: 生成器配置
        batches: 批次列表
        data_config: 数据生成配置
        output_dir: 输出目录
        logger: 日志记录器
    """
    generator = None  # 添加生成器引用以便清理
    
    if not generator_config.get('enabled', False):
        logger.info(f"Generator {generator_config['model_name']} is disabled, skipping...")
        return
    
    model_name = generator_config['model_name']
    model_params = generator_config['model_params']
    num_images_per_prompt = data_config['num_images_per_prompt']
    
    # 创建生成器
    try:
        generator = BaseGenerator.create(generator_type, model_name, model_params)
        logger.info(f"Created {generator_type} generator with model {model_name}")
    except ValueError as e:
        logger.error(f"Failed to create {generator_type} generator: {e}")
        return
    
    # 加载模型
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator.load_model(device)
        logger.info(f"Loaded {generator_type} model on device {device}")
    except Exception as e:
        logger.error(f"Failed to load {generator_type} model: {e}")
        return
    
    # 处理批次
    processed_batches = 0
    total_images = 0
    total_batches = len(batches)
    
    logger.info(f"Processing {generator_type} with {total_batches} batches")
    
    for i, batch in enumerate(batches):
        try:
            batch_id = batch['id']
            progress_percent = ((i + 1) / total_batches) * 100
            logger.info(f"Processing {generator_type} batch {batch_id} ({i + 1}/{total_batches}, {progress_percent:.1f}%)")
            
            # 提取文本提示
            prompts = batch['data']
            prompts = [item['prompt'] for item in prompts]
            
            # 生成图像
            try:
                start_time = time.time()
                images = generator.generate(prompts, device)
                elapsed_time = time.time() - start_time
                total_images += len(images)
                processed_batches += 1
                
                # 保存图像
                save_images(images, batch, output_dir, generator_type, num_images_per_prompt)
                
                logger.info(f"{generator_type} completed batch {batch_id}, "
                           f"generated {len(images)} images in {elapsed_time:.2f}s "
                           f"({i + 1}/{total_batches}, {progress_percent:.1f}%)")
            except Exception as e:
                logger.error(f"Error generating images with {generator_type}, batch {batch_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error processing batch in {generator_type}: {e}")
    
    logger.info("=" * 50)
    logger.info(f"{generator_type} GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total processed batches: {processed_batches}")
    logger.info(f"Total generated images: {total_images}")
    logger.info("=" * 50)
    
    # 清理模型资源
    try:
        if generator is not None:
            generator.clear_model()
            logger.info(f"Cleared model resources for {generator_type}")
    except Exception as e:
        logger.warning(f"Failed to clear model resources for {generator_type}: {e}")
    
    # 清理GPU内存
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Emptied CUDA cache for {generator_type}")
    except Exception as e:
        logger.warning(f"Failed to empty CUDA cache for {generator_type}: {e}")


def process_generator_with_accelerate(generator_type: str, generator_config: Dict[str, Any], 
                                      batches: List[Dict[str, Any]], data_config: Dict[str, Any], 
                                      output_dir: str, logger: logging.Logger):
    """
    使用Accelerate处理生成器的图像生成（支持多卡）
    
    Args:
        generator_type: 生成器类型
        generator_config: 生成器配置
        batches: 批次列表
        data_config: 数据生成配置
        output_dir: 输出目录
        logger: 日志记录器
    """
    generator = None  # 添加生成器引用以便清理
    
    if not generator_config.get('enabled', False):
        try:
            accelerator = Accelerator()
            if accelerator.is_main_process:
                logger.info(f"Generator {generator_type} is disabled, skipping...")
        except:
            logger.info(f"Generator {generator_type} is disabled, skipping...")
        return
    
    # 检查是否安装了accelerate库
    if not ACCELERATE_AVAILABLE:
        try:
            accelerator = Accelerator()
            if accelerator.is_main_process:
                logger.warning("Accelerate library not available. Using single GPU processing.")
        except:
            logger.warning("Accelerate library not available. Using single GPU processing.")
        # 回退到单GPU处理
        return process_generator_single_gpu(generator_type, generator_config, batches, data_config, output_dir, logger)
    
    model_name = generator_config['model_name']
    model_params = generator_config['model_params']
    pipline_type = generator_config['pipline_type']
    num_images_per_prompt = data_config['num_images_per_prompt']
    
    # 初始化Accelerator
    accelerator = Accelerator()
    
    # 创建生成器
    try:
        generator = BaseGenerator.create(pipline_type, model_name, model_params)
        if accelerator.is_main_process:
            logger.info(f"Created {generator_type} generator with model {model_name}")
    except ValueError as e:
        if accelerator.is_main_process:
            logger.error(f"Failed to create {generator_type} generator: {e}")
        return
    
    # 加载模型到Accelerator指定的设备
    try:
        device = accelerator.device
        generator.load_model(device)
        if accelerator.is_main_process:
            logger.info(f"Loaded {generator_type} model on device {device}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Failed to load {generator_type} model: {e}")
        return
    
    # 在所有设备上同步
    accelerator.wait_for_everyone()
    
    # 处理批次 - 使用accelerate的分片功能
    processed_batches = 0
    total_images = 0
    results = []
    
    # 将批次分发到各个进程
    with accelerator.split_between_processes(batches) as batch_shard:
        logger.info(f"Processing {generator_type} with {len(batch_shard)} batches on process {accelerator.process_index}/{accelerator.num_processes}")

        # 计算当前进程的总批次和已完成批次
        total_batches_in_process = len(batch_shard)
        completed_batches_in_process = 0
        
        for i, batch in enumerate(batch_shard):
            try:
                batch_id = batch['id']
                logger.info(f"Processing {generator_type} batch {batch_id} on process {accelerator.process_index}/{accelerator.num_processes} ({i+1}/{total_batches_in_process})")
                
                # 提取文本提示
                prompts = batch['data']
                prompts = [item['prompt'] for item in prompts]
                
                # 生成图像
                try:
                    start_time = time.time()
                    images = generator.generate(prompts, device)
                    elapsed_time = time.time() - start_time
                    total_images += len(images)
                    processed_batches += 1
                    completed_batches_in_process += 1

                    # 保存图像
                    save_images(images, batch, output_dir, generator_type, num_images_per_prompt)
                    
                    logger.info(f"{generator_type} completed batch {batch_id} on process {accelerator.process_index}/{accelerator.num_processes}, "
                               f"generated {len(images)} images in {elapsed_time:.2f}s "
                               f"({completed_batches_in_process}/{total_batches_in_process}, {100*completed_batches_in_process/total_batches_in_process:.1f}%)")

                    results.append({
                        'batch_id': batch_id,
                        'images_count': len(images),
                        'elapsed_time': elapsed_time,
                        'process_index': accelerator.process_index
                    })
                except Exception as e:
                    logger.error(f"Error generating images with {generator_type} on process {accelerator.process_index}/{accelerator.num_processes}, batch {batch_id}: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing batch in {generator_type} on process {accelerator.process_index}/{accelerator.num_processes}: {e}")

            if i < len(batches)// accelerator.num_processes :
                # 在每个batch完成后进行torch同步，确保不同进程的进度一致
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logger.info(f"Completed processing batch {batch_id}, synchronized all processes")

    
    # 收集所有进程的结果
    gathered_results = gather_object(results)
    
    # 输出结果统计
    if accelerator.is_main_process:
        total_processed_batches = len(gathered_results)
        total_images = sum(result['images_count'] for result in gathered_results)
        total_time = sum(result['elapsed_time'] for result in gathered_results)
        
        logger.info("=" * 50)
        logger.info(f"{generator_type} GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total processed batches: {total_processed_batches}")
        logger.info(f"Total generated images: {total_images}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per batch: {total_time/total_processed_batches:.2f}s" if total_processed_batches > 0 else "Average time per batch: N/A")
        logger.info("=" * 50)
    
    # 清理GPU内存和模型资源
    try:
        if generator is not None:
            generator.clear_model()
            if accelerator.is_main_process:
                logger.info(f"Cleared model resources for {generator_type}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning(f"Failed to clear model resources for {generator_type}: {e}")
    
    # 清理GPU内存
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                logger.info(f"Emptied CUDA cache for {generator_type}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning(f"Failed to empty CUDA cache for {generator_type}: {e}")
    
    # 在所有设备上同步
    accelerator.wait_for_everyone()


def process_generator(generator_type: str, generator_config: Dict[str, Any], 
                      prompts: List[Dict[str, str]], data_config: Dict[str, Any], 
                      output_dir: str, logger: logging.Logger):
    """
    处理单个生成器的图像生成（支持多卡）
    
    Args:
        generator_type: 生成器类型
        generator_config: 生成器配置
        prompts: 文本描述列表
        data_config: 数据生成配置
        output_dir: 输出目录
        logger: 日志记录器
    """
    if not generator_config.get('enabled', False):
        # 只有主进程输出日志
        try:
            accelerator = Accelerator()
            if accelerator.is_main_process:
                logger.info(f"Generator {generator_type} is disabled, skipping...")
        except:
            logger.info(f"Generator {generator_type} is disabled, skipping...")
        return

    # 过滤掉已经生成的图像
    prompts = [ prompt for prompt in prompts if not os.path.exists(os.path.join(output_dir, generator_type, prompt['image']))]
    logger.info(f"过滤后剩余 {len(prompts)} 个文本描述待处理")
    if len(prompts) == 0:
        return
    batch_size = generator_config['batch_size']
    
    # 分批处理
    batches = batch_prompts(prompts, batch_size)
    
    # 只有主进程输出日志
    try:
        accelerator = Accelerator()
        if accelerator.is_main_process:
            logger.info(f"Processing with {generator_type} in {len(batches)} batches of size {batch_size}")
    except:
        logger.info(f"Processing with {generator_type} in {len(batches)} batches of size {batch_size}")
    
    # 使用Accelerate处理生成器（如果可用）
    if ACCELERATE_AVAILABLE:
        process_generator_with_accelerate(generator_type, generator_config, batches, data_config, output_dir, logger)
    else:
        # 回退到单GPU处理
        process_generator_single_gpu(generator_type, generator_config, batches, data_config, output_dir, logger)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Generate images from text prompts')
    parser.add_argument('--config', type=str, default='./dataset_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # 初始化Accelerator（用于获取进程信息）
    try:
        accelerator = Accelerator()
    except:
        accelerator = None

    # 原始命令行生成流程
    # 加载配置
    config = load_config(args.config)

    # 设置日志
    logger = setup_logging(config['data_generation']['logging'])
    
    # 输出进程信息
    if accelerator and accelerator.is_main_process:
        logger.info("开始图像生成过程")
        logger.info(f"Running with {accelerator.num_processes} processes")
    elif accelerator is None:
        logger.info("开始图像生成过程（单进程模式）")
    else:
        logger.info(f"Process {accelerator.process_index} started")

    # 获取配置参数
    generators_config = config['generators']
    data_config = config['data_generation']
    output_dir = data_config['output_dir']

    # 加载文本描述
    input_file = data_config['input_file']
    prompts = load_prompts(input_file)


    if accelerator is None or accelerator.is_main_process:
        logger.info(f"从 {input_file} 加载了 {len(prompts)} 个文本描述")

    # 记录开始时间
    start_time = time.time()
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"开始生成")

    # 为每个启用的生成器处理数据
    for generator_type, generator_config in generators_config.items():
        process_generator(generator_type, generator_config, prompts, data_config, output_dir, logger)

    # 计算总耗时
    elapsed_time = time.time() - start_time
    if accelerator is None or accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("ALL GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
