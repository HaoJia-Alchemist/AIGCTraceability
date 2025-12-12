import argparse
import os
import glob
import json
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import DATASETS_FACTORY
from models import make_model
from utils.metrics import euclidean_distance


class GalleryDataset(Dataset):
    """
    Dataset class for gallery images to support batch processing
    """
    def __init__(self, gallery_images, transform=None):
        self.gallery_images = gallery_images
        self.transform = transform
        
    def __len__(self):
        return len(self.gallery_images)
        
    def __getitem__(self, idx):
        img_path, df_id = self.gallery_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, df_id, img_path
        except Exception as e:
            print(f"Warning: Could not process image {img_path}: {str(e)}")
            # Return a zero tensor if image loading fails
            if self.transform:
                image = Image.new('RGB', (224, 224))  # Create a dummy image
                image = self.transform(image)
            else:
                image = torch.zeros(3, 224, 224)
            return image, df_id, img_path


def scan_gallery_images(gallery_dir):
    """
    Scan gallery images from directory structure
    Expected structure: gallery_dir/method_name/*.jpg
    """
    gallery_images = []
    df_id_begin = 0
    df_id_name_map = {}
    df_name_id_map = {}
    
    # 遍历gallery目录下的所有子目录（方法名）
    method_dirs = [d for d in os.listdir(gallery_dir) 
                   if os.path.isdir(os.path.join(gallery_dir, d))]
    
    for method_name in method_dirs:
        method_path = os.path.join(gallery_dir, method_name)
        # 获取该方法目录下的所有图像文件
        image_files = glob.glob(os.path.join(method_path, "*"))
        image_files = [f for f in image_files if os.path.isfile(f) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_path in image_files:
            # 为每个方法分配唯一的ID
            if method_name not in df_name_id_map:
                df_id_name_map[df_id_begin] = method_name
                df_name_id_map[method_name] = df_id_begin
                df_id_begin += 1
            df_id = df_name_id_map[method_name]
            gallery_images.append((img_path, df_id))  # 空字符串是prompt占位符
    
    return gallery_images, df_id_name_map, df_name_id_map


def extract_features_from_images_batch(model, gallery_images, transform, device, batch_size=32):
    """
    Extract features from gallery images using batching
    """
    model.eval()
    gallery_dataset = GalleryDataset(gallery_images, transform=transform)
    dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    features = []
    ids = []
    paths = []
    
    with torch.no_grad():
        for batch_images, batch_ids, batch_paths in tqdm(dataloader):
            batch_images = batch_images.to(device)
            batch_features = model(batch_images)
            if isinstance(batch_features, (tuple, list)):
                batch_features = batch_features[0]
            batch_features = F.normalize(batch_features, p=2, dim=1)
            
            features.append(batch_features.cpu())
            ids.extend(batch_ids.tolist())
            paths.extend(batch_paths)
    
    if features:
        features = torch.cat(features, dim=0)
    
    return features, ids, paths


def extract_features_from_images(model, gallery_images, transform, device):
    """
    Extract features from gallery images
    """
    model.eval()
    features = []
    ids = []
    paths = []
    
    with torch.no_grad():
        for img_path, img_id in gallery_images:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                feature = model(image_tensor)
                if isinstance(feature, (tuple, list)):
                    feature = feature[0]
                feature = F.normalize(feature, p=2, dim=1)
                features.append(feature.cpu())
                ids.append(img_id)
                paths.append(img_path)
            except Exception as e:
                print(f"Warning: Could not process image {img_path}: {str(e)}")
                
    if features:
        features = torch.cat(features, dim=0)
    
    return features, ids, paths


def save_gallery_features(gallery_feats, gallery_ids, gallery_paths, save_path):
    """
    Save gallery features to npz file
    """
    np.savez_compressed(
        save_path,
        gallery_feats=gallery_feats.numpy(),
        gallery_ids=gallery_ids,
        gallery_paths=gallery_paths
    )

def get_data_transforms(config):
    trans = T.Compose([
        T.Resize(config['resolution']),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])
    return trans

def extract_feature(model, image, transform):
    """
    Extract feature from a single image
    """
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(next(model.parameters()).device)
        feature = model(image)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        feature = F.normalize(feature, p=2, dim=1)
        return feature.cpu()


def trace_image(config_file, model_weights, gallery_file, query_image_path, top_k=10, force_rebuild=False):
    """
    Trace a single query image against gallery
    
    Args:
        config_file: Path to model config YAML file
        model_weights: Path to trained model weights
        gallery_file: Path to gallery features file (.npz) or gallery directory
        query_image_path: Path to query image
        top_k: Number of top results to return (default: 10)
        force_rebuild: Force rebuild gallery features even if .npz file exists
    
    Returns:
        List of top-k results with ids and paths
    """
    # Load config
    config = OmegaConf.load(config_file)

    # Prepare model
    model = make_model(config, num_classes=15)  # num_classes doesn't matter for inference
    model = model.float()
    
    # Load model weights
    state_dict = torch.load(model_weights, map_location='cpu')
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    # Get transforms
    transform = get_data_transforms(config['dataset'])

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Need to extract features from images
    print("Extracting gallery features from images...")
    gallery_images, df_id_name_map, df_name_id_map = scan_gallery_images(gallery_file)

    batch_size = config['dataset']['test_batch_size']
    # Extract features
    if batch_size > 1:
        gallery_feats, gallery_ids, gallery_paths = extract_features_from_images_batch(
            model, gallery_images, transform, device, batch_size)
    else:
        gallery_feats, gallery_ids, gallery_paths = extract_features_from_images(
            model, gallery_images, transform, device)

    
    gallery_feats = gallery_feats.to(device)

    # Load and preprocess query image
    query_image = Image.open(query_image_path).convert('RGB')

    # Extract query feature
    query_feat = extract_feature(model, query_image, transform)
    query_feat = query_feat.to(device)

    # Calculate distances
    distmat = euclidean_distance(query_feat, gallery_feats)

    # Get top-k results
    indices = np.argsort(distmat, axis=1)[0][:top_k]
    
    results = []
    for i, idx in enumerate(indices):
        method_name = df_id_name_map.get(gallery_ids[idx], gallery_ids[idx]) if df_id_name_map else gallery_ids[idx]
        results.append({
            'rank': i + 1,
            'id': gallery_ids[idx],
            'method': method_name,
            'path': gallery_paths[idx],
            'distance': float(distmat[0][idx])
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Trace a single query image')
    parser.add_argument('--config_file', default="config/configs_dir/effort.yaml", type=str, help='Path to config YAML file')
    parser.add_argument('--model_weights', default="/home/jh/disk/workspace/AIGCTraceability/DeepfakeTraceability/logs/effort_train_20251104204224/best_model/model.pth", type=str, help='Path to model weights')
    parser.add_argument('--gallery_file', default="/home/jh/disk/datasets/AIGCTraceability/demo_gallery", type=str, help='Path to gallery directory')
    parser.add_argument('--query_image', default="/home/jh/disk/datasets/AIGCTraceability/demo_query/DMD_140430106_2978fda105.jpg", type=str, help='Path to query image')
    parser.add_argument('--top_k', default=5, type=int,  help='Number of top results to return')

    args = parser.parse_args()
    
    results = trace_image(
        config_file=args.config_file,
        model_weights=args.model_weights,
        gallery_file=args.gallery_file,
        query_image_path=args.query_image,
        top_k=args.top_k,
    )
    
    print("Tracing Results:")
    print("-" * 50)
    for result in results:
        print(f"Rank {result['rank']:2d}: Method={result['method']:<15} "
              f"Distance={result['distance']:.4f} Path={result['path']}")


if __name__ == '__main__':
    main()