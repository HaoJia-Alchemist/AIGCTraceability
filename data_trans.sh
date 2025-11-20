#!/bin/bash

# 服务器信息（无需修改）
SERVER="172.16.150.201"
BASE_TARGET="/raid/NFSshare/datasets2/"

# 数据集列表：每行格式为 "本地源路径 目标路径后缀"
# 示例："/home/ywl/disk1/Datasets/thumos14 TemporalActionDetection"
DATASETS=(
    # 鲁玉菲
    "/home/lyf/disk1/datasets/Celeb-reID-light ReID"
    "/home/lyf/disk1/datasets/DATA_FOLDER/flowers102 ReID"
)

# 循环迁移每个数据集
for item in "${DATASETS[@]}"; do
    # 解析源路径和目标后缀
    SRC=$(echo "$item" | awk '{print $1}')
    DST_SUFFIX=$(echo "$item" | awk '{print $2}')
    TARGET="${SERVER}:${BASE_TARGET}${DST_SUFFIX}"

    echo "====================================="
    echo "开始迁移数据集："
    echo "本地源路径：$SRC"
    echo "目标服务器路径：$TARGET"
    echo "====================================="

    # 执行rsync命令（带断点续传和进度显示）
    rsync -a --partial "$SRC" "$TARGET"

    # 检查传输是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 数据集迁移成功：$SRC"
    else
        echo "❌ 数据集迁移失败：$SRC"
    fi
    echo -e "\n"
done

echo "所有数据集迁移任务已处理完毕"
