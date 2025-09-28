# AIGC图像溯源数据集生成平台

这是一个用于生成AIGC图像溯源数据集的工具平台，支持多种生成器并提供可视化管理界面。

## 功能特点

1. **多种生成器支持**：
   - Stable Diffusion v1.5
   - FLUX.1
   - Imagen (占位支持)

2. **可视化Web界面**：
   - 配置管理
   - 提示管理
   - 任务管理
   - 实时进度监控

3. **多GPU支持**：
   - 支持在多个GPU上并行处理
   - 负载均衡分配

4. **灵活的配置**：
   - YAML配置文件
   - 可调整的批处理大小
   - 可配置的模型参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行模式

```bash
cd src
python generate_data.py --config dataset_config.yaml
```

### Web服务模式

```bash
cd src
python generate_data.py --web
```

然后在浏览器中访问 http://localhost:5000

## Web界面功能

### 1. 首页
展示平台功能概览和快速导航入口。

### 2. 配置管理
- 管理数据生成配置（输入文件、输出目录、GPU设置等）
- 管理生成器配置（启用/禁用、模型名称、批处理大小、模型参数等）

### 3. 提示管理
- 查看、添加、编辑、删除文本提示
- 支持分页浏览大量提示
- 保存提示到数据库

### 4. 任务管理
- 创建新的生成任务
- 查看任务状态和进度
- 实时监控任务执行情况

## 配置文件说明

配置文件采用YAML格式，包含以下主要部分：

```yaml
# 数据生成配置
data_generation:
  input_file: "./caption_combined.txt"    # 输入文本文件路径
  output_dir: "/path/to/output"           # 输出目录
  gpu_ids: [0, 1]                         # 使用的GPU ID列表
  num_images_per_prompt: 1                # 每个提示生成图像数量
  logging:                                # 日志配置
    level: "INFO"
    file: "logs/generation.log"
    console_output: true

# 生成器配置
generators:
  StableDiffusionV1_5:
    enabled: true
    model_name: "sd-legacy/stable-diffusion-v1-5"
    model_params:
      guidance_scale: 7.5
      num_inference_steps: 50
    batch_size: 16
```

## 开发说明

项目结构：
```
src/
├── generate_data.py          # 命令行入口和核心生成逻辑
├── web_app.py                # Web服务入口
├── dataset_config.yaml       # 配置文件
├── caption_combined.txt      # 示例文本提示文件
├── generators/               # 生成器实现
│   ├── base_generator.py
│   ├── stable_diffusion_v1_5_generator.py
│   └── ...
├── templates/                # Web界面模板
│   ├── base.html
│   ├── index.html
│   ├── config.html
│   ├── prompts.html
│   └── tasks.html
└── static/                   # 静态资源文件（如CSS、JS等）
```

## 注意事项

1. 首次运行时会自动创建SQLite数据库用于存储提示和任务信息
2. 文本提示优先从数据库加载，如果数据库为空则从文件加载
3. Web服务默认监听5000端口
4. 生成的图像将按照配置保存到指定输出目录