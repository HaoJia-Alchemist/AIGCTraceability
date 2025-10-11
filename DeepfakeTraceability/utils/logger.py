import os
import logging
from logging.handlers import RotatingFileHandler
import time
from accelerate import Accelerator

class RankFilter(logging.Filter):
    def __init__(self, process_index, accelerator):
        super().__init__()
        self.process_index = process_index
        self.accelerator = accelerator


    def filter(self, record):
        return self.accelerator.process_index == self.process_index

def create_logger(config):
    """
        创建日志记录器，根据进程索引过滤日志输出

    :param config: 日志配置字典，包含日志级别、日志目录等
    :param accelerator: Accelerator 实例，用于获取当前进程索引
    :return: 配置好的日志记录器
    """
    accelerator = Accelerator()
    # 创建日志目录
    log_file = config["log_file"]
    log_dir = config["log_dir"]
    if log_dir and os.path.exists(log_dir) == False and accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger('Train')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))

    # 清除现有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器（支持轮转）
    if accelerator.is_main_process:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台处理器
    if config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 进程过滤器
    rank_filter = RankFilter(0, accelerator)
    logger.addFilter(rank_filter)

    # 日志创建完成
    logger.info(f"Logger created with log file: {log_file}")

    return logger