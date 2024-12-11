import torch
from torch.utils.data import DataLoader
from groundingdino.models import build_model
from groundingdino.util.misc import NestedTensor
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.slconfig import SLConfig
import argparse
import torch.nn.functional as F
import os
from datasets import GroundingDINODataset, collate_fn
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', '-p', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='outputs', type=str)
    return parser
def xywh2xyxy(x):
    """
    x: tensor shape [..., 4] - [x_center, y_center, width, height]
    return: tensor shape [..., 4] - [x1, y1, x2, y2] (左上角和右下角坐标)
    """
    x_c, y_c, w, h = x.unbind(-1)  # 拆分最后一维
    
    # 计算左上角和右下角坐标
    x1 = x_c - 0.5 * w  # 左上角x
    y1 = y_c - 0.5 * h  # 左上角y
    x2 = x_c + 0.5 * w  # 右下角x
    y2 = y_c + 0.5 * h  # 右下角y
    
    return torch.stack([x1, y1, x2, y2], dim=-1)
# 计算IoU
def box_iou(box1, box2):
    
    # 转换为xyxy格式
    box1_xyxy = xywh2xyxy(box1.unsqueeze(0))  # [1, 4]
    box2_xyxy = xywh2xyxy(box2.unsqueeze(0))  # [1, 4]
    
    # 计算交集区域
    inter_x1 = torch.max(box1_xyxy[..., 0], box2_xyxy[..., 0])
    inter_y1 = torch.max(box1_xyxy[..., 1], box2_xyxy[..., 1])
    inter_x2 = torch.min(box1_xyxy[..., 2], box2_xyxy[..., 2])
    inter_y2 = torch.min(box1_xyxy[..., 3], box2_xyxy[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算各自面积
    box1_area = (box1_xyxy[..., 2] - box1_xyxy[..., 0]) * \
                (box1_xyxy[..., 3] - box1_xyxy[..., 1])
    box2_area = (box2_xyxy[..., 2] - box2_xyxy[..., 0]) * \
                (box2_xyxy[..., 3] - box2_xyxy[..., 1])
    
    # 计算并集面积
    union = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union
    
    return iou
def box_giou(box1, box2):
    iou = box_iou(box1, box2)
    
    # 计算外接矩形
    box1_xyxy = xywh2xyxy(box1.unsqueeze(0))
    box2_xyxy = xywh2xyxy(box2.unsqueeze(0))
    
    inter_x1 = torch.max(box1_xyxy[..., 0], box2_xyxy[..., 0])
    inter_y1 = torch.max(box1_xyxy[..., 1], box2_xyxy[..., 1])
    inter_x2 = torch.min(box1_xyxy[..., 2], box2_xyxy[..., 2])
    inter_y2 = torch.min(box1_xyxy[..., 3], box2_xyxy[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算各自面积
    box1_area = (box1_xyxy[..., 2] - box1_xyxy[..., 0]) * \
                (box1_xyxy[..., 3] - box1_xyxy[..., 1])
    box2_area = (box2_xyxy[..., 2] - box2_xyxy[..., 0]) * \
                (box2_xyxy[..., 3] - box2_xyxy[..., 1])
    
    # 计算并集面积
    union = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union
    
    # 计算最小外接矩形(闭包)的坐标
    enclose_x1 = torch.min(box1_xyxy[..., 0], box2_xyxy[..., 0])
    enclose_y1 = torch.min(box1_xyxy[..., 1], box2_xyxy[..., 1])
    enclose_x2 = torch.max(box1_xyxy[..., 2], box2_xyxy[..., 2])
    enclose_y2 = torch.max(box1_xyxy[..., 3], box2_xyxy[..., 3])
    
    # 计算最小外接矩形的面积
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # 计算GIoU
    giou = iou - (enclose_area - union) / enclose_area
    
    return giou
def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    

    for batch_idx, (images, targets) in enumerate(data_loader):
        # 移动数据到设备
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(images, targets)
        
        # 获取预测框和置信度
        pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]    # [batch_size, num_queries, 4]
        
        batch_size = len(targets)
        losses = []
        for b in range(batch_size):
            # 获取置信度最高的预测框
            confidence = pred_logits[b].softmax(-1)
            max_conf, _ = confidence.max(-1)
            best_query_idx = max_conf.argmax()
            
            # 获取最佳预测框和对应的ground truth
            pred_box = pred_boxes[b, best_query_idx]  # [4]
            gt_box = targets[b]['boxes'][0]          # [4]
            
            
            iou = box_iou(pred_box, gt_box)
            # Confidence Loss:
            # 1. 对于高IoU的预测框,增加其confidence
            # 2. 对于低IoU的预测框,降低其confidence
            iou_threshold = 0.5
            pos_mask = iou > iou_threshold
            neg_mask = iou <= iou_threshold
            

            
            if iou > iou_threshold:
                # 正样本: 让confidence接近IoU值
                confidence_loss = F.mse_loss(max_conf[best_query_idx], iou)
            else:
                # 负样本: 让confidence接近0
                confidence_loss = max_conf[best_query_idx] ** 2
            
            
            # 计算GIoU loss as part of the loss
            
            loss_giou = 1 - box_giou(pred_box, gt_box)

            losses.append(loss_giou+confidence_loss)
            
            # 更新评估指标
            # total_iou += iou.item()

        
        # 计算batch的平均损失
        loss = torch.stack(losses).mean()
        confidence_losses = confidence_loss.mean()
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(data_loader)

def val(model,data_loader,device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_giou = 0
    total_confidence_loss = 0
    total_confidence_losses = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        outputs = model(images,targets)
        pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]    # [batch_size, num_queries, 4]
        
        batch_size = len(targets)
        losses = []
        giou_losses=[]
        for b in range(batch_size):
            # 获取置信度最高的预测框
            confidence = pred_logits[b].softmax(-1)
            max_conf, _ = confidence.max(-1)
            best_query_idx = max_conf.argmax()
            
            # 获取最佳预测框和对应的ground truth
            pred_box = pred_boxes[b, best_query_idx]  # [4]
            gt_box = targets[b]['boxes'][0]          # [4]
            
            
            iou = box_iou(pred_box, gt_box)
            # Confidence Loss:
            # 1. 对于高IoU的预测框,增加其confidence
            # 2. 对于低IoU的预测框,降低其confidence
            iou_threshold = 0.5
            pos_mask = iou > iou_threshold
            neg_mask = iou <= iou_threshold
            

            
            if iou > iou_threshold:
                # 正样本: 让confidence接近IoU值
                confidence_loss = F.mse_loss(max_conf[best_query_idx], iou)
            else:
                # 负样本: 让confidence接近0
                confidence_loss = max_conf[best_query_idx] ** 2
            
            
            # 计算GIoU loss as part of the loss
            
            loss_giou = 1 - box_giou(pred_box, gt_box)

            losses.append(loss_giou+confidence_loss)
            giou_losses.append(loss_giou)
            # 更新评估指标
            # total_iou += iou.item()

        
        # 计算batch的平均损失
        loss = torch.stack(losses).mean()
        giou_losses = torch.stack(giou_losses).mean()
    
    return loss,giou_losses


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_setup = SLConfig.fromfile(args.config_file)
    model_setup.device = args.device
    # Build model
    model = build_model(model_setup)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.train()
    # Move model to device
    model = model.to(args.device)
    
    # Create dataset and dataloader
    # Note: You'll need to implement your custom Dataset class
    train_dataset = GroundingDINODataset(args.train_data_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn  # You'll need to implement this
    )
    val_dataset = GroundingDINODataset(args.train_data_path,False)
    val_loader= DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn  # You'll need to implement this
    )
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch
        )
        
        print(f"Epoch {epoch} finished, loss: {train_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }
            
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f'checkpoint_{epoch:03d}.pth')
            )
            
        # Validation
            val_loss,val_giou_loss = val(model,val_loader,args.device)
            print(f"Epoch {epoch} finished, val_loss: {val_loss:.4f},val_giou_loss: {val_giou_loss:.4f}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GroundingDINO training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)