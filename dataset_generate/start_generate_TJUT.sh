#!/bin/bash

# 从dataset_config.yaml中读取gpu_ids并自动规划CUDA可见设备和num_processes

# 检查yq命令是否存在
if ! command -v yq &> /dev/null
then
    echo "yq命令未找到，请先安装yq: pip install yq"
    exit 1
fi

# 检查配置文件是否存在
CONFIG_FILE="./dataset_config_TJUT.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 从配置文件中读取gpu_ids
GPU_IDS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['data_generation']['gpu_ids'])")
# 检查gpu_ids是否为空或null
if [ "$GPU_IDS" == "null" ] || [ "$GPU_IDS" == "" ]; then
    echo "未配置gpu_ids"
    exit 0
fi
# 计算GPU数量
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')

echo "检测到配置的GPU IDs: $GPU_IDS"
echo "将使用 $NUM_PROCESSES 个进程"

# 设置环境变量并启动加速器
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" generate_data.py --config "$CONFIG_FILE"