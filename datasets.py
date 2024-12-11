import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from groundingdino.util.misc import NestedTensor,nested_tensor_from_tensor_list
from torchvision import transforms
class GroundingDINODataset(Dataset):
    def __init__(self, data_path,tv:bool=True):
        '''
        data_path: data path
        tv: True for train, False for val
        '''
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, "image")
        
        # 加载标注文件
        if(tv):
            with open(os.path.join(data_path, "annotations_train.json"), 'r') as f:
                self.annotations = json.load(f)
        else:
            with open(os.path.join(data_path, "annotations_val.json"), 'r') as f:
                self.annotations = json.load(f)
        count = 0
        # 创建image_id到图像信息的映射
        self.image_info = {}
        for img in self.annotations['images']:
            # self.image_info[img['id']] = img
            self.image_info[count] = img
            count+=1
        count =0
        # 创建image_id到标注的映射
        self.image_annotations = {}
        for ann in self.annotations['annotations']:
            # self.image_annotations[ann['image_id']] = ann
            self.image_annotations[count] = ann
            count+=1
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __getitem__(self, idx):
        # 获取图像信息
        image_info = self.image_info[idx]
        img_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # 加载图像
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 获取bbox标注
        ann = self.image_annotations[idx]
        x_min, y_min, w, h = ann['bbox']
        
        # 转换为归一化的[center_x, center_y, width, height]格式
        center_x = (x_min + w/2) / img_width    # 中心点x坐标归一化
        center_y = (y_min + h/2) / img_height   # 中心点y坐标归一化
        norm_w = w / img_width                   # 宽度归一化
        norm_h = h / img_height                  # 高度归一化
        
        # 创建target字典
        target = {
            'boxes': torch.tensor([[center_x, center_y, norm_w, norm_h]], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id']], dtype=torch.long),
            'image_id': torch.tensor([ann['image_id']]),
            'caption': "polyp.",
            'orig_size': torch.tensor([img_height, img_width]),
            'size': torch.tensor([img_height, img_width]),
            
            # 保存原始bbox以便验证
            'orig_bbox': torch.tensor([x_min, y_min, w, h])
        }
        
        return image, target
    def __len__(self):
        return len(self.annotations['images'])


def collate_fn(batch):
    images = []
    targets = []
    
    # 收集原始尺寸信息
    original_sizes = []
    
    for img, target in batch:
        # 保存原始尺寸
        w, h = img.size
        original_sizes.append((h, w))
        
        # 转换图像为tensor
        img_tensor = transforms.ToTensor()(img)
        images.append(img_tensor)
        targets.append(target)
    
    # 创建NestedTensor
    batched_images = nested_tensor_from_tensor_list(images)
    
    # 打印调试信息
    # print(f"\nBatch info:")
    # print(f"Number of images: {len(images)}")
    # print(f"Original sizes: {original_sizes}")
    # print(f"Batched tensor shape: {batched_images.tensors.shape}")
    # print(f"Mask shape: {batched_images.mask.shape}")
    
    # 在target中添加原始尺寸信息
    for target, size in zip(targets, original_sizes):
        target['orig_size'] = torch.tensor(size)
    
    return batched_images, targets