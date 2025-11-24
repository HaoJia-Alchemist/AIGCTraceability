import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from accelerate import Accelerator


def create_logger(config):
    """
    创建日志记录器。
    策略：只有主进程(Rank 0)会被赋予Handler（控制台和文件），
    从而保证只有主进程会进行实际的打印操作。
    """
    accelerator = Accelerator()

    # 1. 获取根日志记录器
    # 使用 getLogger() 不带名字获取的是 root logger，
    # 这样后续代码中使用 logging.getLogger(__name__) 都会继承这里的设置
    logger = logging.getLogger()

    # 2. 清除现有的处理器 (防止多次调用导致重复打印)
    logger.handlers.clear()

    # 3. 设置日志级别
    # 如果是主进程，使用配置的级别；否则设置为 ERROR，减少非主进程的开销
    target_level = getattr(logging, config.get('level', 'INFO').upper())

    if accelerator.is_main_process:
        logger.setLevel(target_level)
    else:
        # 非主进程只记录错误，或者你可以设置为 logging.CRITICAL 以完全静音
        logger.setLevel(logging.ERROR)

    # 4. 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. 仅在主进程添加 Handler
    # 这是解决多进程重复打印最根本、最有效的方法
    if accelerator.is_main_process:
        # --- 文件处理器 ---
        log_file = config.get("log_file")
        log_dir = config.get("log_dir")

        if log_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(target_level)
            logger.addHandler(file_handler)

        # --- 控制台处理器 ---
        if config.get('console_output', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(target_level)
            logger.addHandler(console_handler)

        logger.info(f"Logger initialized on Main Process. Log file: {log_file}")
    else:
        # 非主进程不添加任何 Handler，
        # 因此即使它产生了 ERROR 级别的日志，如果不添加 Handler，默认可能会输出到 stderr (LastResort)
        # 或者直接被吞掉。通常在 DDP 训练中，我们只关心主进程输出。
        pass

    return logger