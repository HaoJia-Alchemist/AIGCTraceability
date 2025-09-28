import csv
import json
import random
from collections import OrderedDict

def split_dataset(csv_file, train_ratio=0.8, test_ratio=0.15, gallery_ratio=0.05):
    """
    读取caption_combined.txt，按照train, test, gallery来划分数据集
    
    Args:
        csv_file: 输入的CSV文件路径
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        gallery_ratio: gallery集比例
    """
    # 读取所有数据
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'image': row['image'],
                'caption': row['caption']
            })
    
    # 打乱数据
    random.seed(42)  # 固定随机种子以确保可重复性
    random.shuffle(data)
    
    # 计算各数据集大小
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    # gallery_size = int(total_size * gallery_ratio)
    
    # 划分数据集
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    gallery_data = data[train_size + test_size:]
    
    # 保存为JSON文件
    with open('train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    with open('gallery.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)
    
    print(f"数据集划分完成:")
    print(f"  总数据量: {total_size}")
    print(f"  训练集: {len(train_data)} ({len(train_data)/total_size*100:.1f}%)")
    print(f"  测试集: {len(test_data)} ({len(test_data)/total_size*100:.1f}%)")
    print(f"  Gallery集: {len(gallery_data)} ({len(gallery_data)/total_size*100:.1f}%)")

if __name__ == "__main__":
    split_dataset('caption_combined.txt')