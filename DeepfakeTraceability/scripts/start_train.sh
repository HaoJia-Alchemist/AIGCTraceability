#cd  /jsj_zyb/jh/workspace/AIGCTraceability/DeepfakeTraceability
#cd  /home/jh/disk/workspace/AIGCTraceability/DeepfakeTraceability
cd  /root/autodl-fs/workspace/GenTrace30/DeepfakeTraceability
#Train_Config_File=./config/train_config.yaml
#Train_Config_File=./config/train_config_SDAI.yaml
Train_Config_File=./config/train_config_autodl.yaml
# 检查yq命令是否存在
if ! command -v yq &> /dev/null
then
    echo "yq命令未找到，请先安装yq: pip install yq"
    exit 1
fi

# 从配置文件中读取gpu_ids

GPU_IDS=$(python -c "import yaml; print(yaml.safe_load(open('$Train_Config_File'))['gpu_ids'])")
# 检查gpu_ids是否为空或null
if [ "$GPU_IDS" == "null" ] || [ "$GPU_IDS" == "" ]; then
    echo "未配置gpu_ids"
    exit 0
fi
# 计算GPU数量
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')

echo "检测到配置的GPU IDs: $GPU_IDS"
echo "将使用 $NUM_PROCESSES 个进程"

# 根据进程数决定是否启用--multi_gpu参数
MULTI_GPU_ARG=""
if [ "$NUM_PROCESSES" -gt 1 ]; then
    MULTI_GPU_ARG="--multi_gpu"
fi

# 设置环境变量并启动加速器（动态添加多GPU参数）
#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/efficientnet.yaml" --train_config_file $Train_Config_File --opt task_name='efficientnet_train' dataset.sampler='softmax_triplet'
#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/resnet50.yaml"
#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/clip_vit_l14.yaml"
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/clip_vit_l14.yaml" --train_config_file $Train_Config_File  --opt task_name='clip_vit_l14_triplet_train' dataset.sampler='softmax_triplet'


#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/effort.yaml" --train_config_file $Train_Config_File  --opt task_name='effort_train' dataset.sampler='softmax_triplet'

#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/prompt_learn.yaml" --train_config_file $Train_Config_File --opt task_name='clip_vit_l14' dataset.sampler='softmax_triplet'

#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/xception.yaml" --train_config_file $Train_Config_File --opt task_name='xception_train' dataset.sampler='softmax_triplet'

#CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch $MULTI_GPU_ARG --num_processes "$NUM_PROCESSES" train.py --config "./config/configs_dir/clip_vit_b16.yaml" --train_config_file $Train_Config_File --opt task_name='clip_vit_b16_train' dataset.sampler='softmax_triplet'