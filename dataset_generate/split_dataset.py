import csv
import json
import os
import random
from collections import OrderedDict

def split_dataset(csv_file, root_dir, df_split_ratio=None, prompt_split_ratio=None):

    if prompt_split_ratio is None:
        prompt_split_ratio = [0.8, 0.15, 0.05]
    if df_split_ratio is None:
        df_split_ratio = [0.5, 0.5]
    df_list = os.listdir(root_dir)
    # 过滤不是文件夹的项
    df_list = [df for df in df_list if os.path.isdir(os.path.join(root_dir, df))]
    # 读取所有数据
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'image': row['image'],
                'prompt': row['prompt'],
            })

    # 打乱数据
    random.seed(42)  # 固定随机种子以确保可重复性
    random.shuffle(data)
    random.shuffle(df_list)

    # -------------------------- 构建"提示词未见过"评估场景的数据结构 --------------------------
    # 场景说明：查询集和画廊集的类别与训练集相同（类别已知），仅提示词未在训练中出现
    # 划分Deepfake方法文件夹：训练集文件夹占比为df_split_ratio[0]，查询/画廊集与训练集文件夹相同
    train_size = int(len(data) * prompt_split_ratio[0])
    query_size = int(len(data) * prompt_split_ratio[1])
    train_data = data[:train_size]
    query_data = data[train_size:train_size + query_size]
    gallery_data = data[train_size + query_size:]
    # df_list 按照df_split_ratio划分，训练集在评估方法1中站df_split_ratio[0],query、gallery在评估方法1中与训练集相同
    train_df_list = df_list[:int(len(df_list) * df_split_ratio[0])] # 训练集类别文件夹
    query_df_list = df_list[:int(len(df_list) * df_split_ratio[0])] # 查询集类别文件夹（与训练集相同）
    gallery_df_list = df_list[:int(len(df_list) * df_split_ratio[0])] # 画廊集类别文件夹（与查询集相同）
    prompt_unseen_evaluation = OrderedDict()
    prompt_unseen_evaluation['train'] = {}  # 训练集：key为类别文件夹，value为训练数据
    prompt_unseen_evaluation['query'] = {}  # 查询集：key为类别文件夹，value为查询数据
    prompt_unseen_evaluation['gallery'] = {}  # 画廊集：key为类别文件夹，value为画廊数据

    # 为每个类别文件夹绑定对应数据集
    for df_name in train_df_list:
        prompt_unseen_evaluation['train'][df_name] = train_data
    for df_name in query_df_list:
        prompt_unseen_evaluation['query'][df_name] = query_data
    for df_name in gallery_df_list:
        prompt_unseen_evaluation['gallery'][df_name] = gallery_data

    # -------------------------- 构建"类别未见过"评估场景的数据结构 --------------------------
    # 场景说明：查询集和画廊集的类别与训练集完全不同（类别未知）
    # 重新划分类别文件夹：训练集与查询/画廊集文件夹无重叠
    train_df_list = df_list[:int(len(df_list) * df_split_ratio[0])]  # 训练集类别文件夹
    query_df_list = df_list[int(len(df_list) * df_split_ratio[0]):]  # 查询集类别文件夹（与训练集无重叠）
    gallery_df_list = query_df_list  # 画廊集类别文件夹（与查询集相同）

    # 创建有序字典存储该场景的划分结果
    unseen_evaluation = OrderedDict()
    unseen_evaluation['train'] = {}  # 训练集：key为类别文件夹，value为训练数据
    unseen_evaluation['query'] = {}  # 查询集：key为类别文件夹，value为查询数据
    unseen_evaluation['gallery'] = {}  # 画廊集：key为类别文件夹，value为画廊数据

    # 为每个类别文件夹绑定对应数据集
    for df_name in train_df_list:
        unseen_evaluation['train'][df_name] = train_data
    for df_name in query_df_list:
        unseen_evaluation['query'][df_name] = query_data
    for df_name in gallery_df_list:
        unseen_evaluation['gallery'][df_name] = gallery_data

    # 将两种评估场景的划分结果保存为JSON文件（用于后续训练和评估）
    with open('prompt_unseen_evaluation.json', 'w') as f:
        json.dump(prompt_unseen_evaluation, f)
    with open('unseen_evaluation.json', 'w') as f:
            json.dump(unseen_evaluation, f)

if __name__ == "__main__":
    root = "/home/jh/disk/datasets/AIGCTraceability/DFT30"
    split_dataset('caption_combined.txt', root_dir=root, df_split_ratio=[0.5,0.5], prompt_split_ratio=[0.8,0.15,0.05])