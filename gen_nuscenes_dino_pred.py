#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, warnings
import torch
import numpy as np
from tqdm import tqdm

# GroundingDINO inference helpers
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

# 與 GLIP 輸出一致的資料結構
from maskrcnn_benchmark.structures.bounding_box import BoxList


# ---- 針對 nuScenes 10 類的常見關鍵字（可依需要擴充）----
DEFAULT_SYNONYMS = {
    "car": ["car","vehicle","sedan","van","taxi","pickup"],
    "truck": ["truck","lorry"],
    "construction_vehicle": ["construction vehicle","excavator","bulldozer","cement mixer","tractor","crane","grader","digger"],
    "bus": ["bus","coach","minibus"],
    "trailer": ["trailer","semi trailer","semi-trailer"],
    "barrier": ["barrier","road barrier","guard rail","guardrail","fence","barricade","traffic barrier","rail","roadblock"],
    "motorcycle": ["motorcycle","motorbike","moto","scooter"],
    "bicycle": ["bicycle","bike","cyclist","biker","bicyclist"],
    "pedestrian": ["pedestrian","person","people","man","woman","runner","walker"],
    "traffic_cone": ["traffic cone","cone","pylon"]
}


def build_class_maps_from_categories(categories):
    """
    從 coco.json 的 categories 建立：
    - id -> name 的索引
    - name -> id 的索引
    並生成一份 synonyms（若 name 在 DEFAULT_SYNONYMS 內則帶入，否則只用 name 本身）
    """
    id2name = {}
    name2id = {}
    synonyms = {}
    for cat in categories:
        cid = int(cat["id"])
        name = str(cat["name"])
        id2name[cid] = name
        name2id[name] = cid
        if name in DEFAULT_SYNONYMS:
            synonyms[name] = [name] + DEFAULT_SYNONYMS[name]
        else:
            synonyms[name] = [name]
    return id2name, name2id, synonyms


def build_prompt_from_categories(id2name):
    """
    GroundingDINO 推論用的文字 prompt：以句點分隔，盡量貼近你的類別命名。
    e.g. "car . truck . construction vehicle . ..."
    """
    # 依類別 id 排序，讓順序可控（與 labels id 無強制關係，但可減少混淆）
    names_sorted = [id2name[i] for i in sorted(id2name.keys())]
    # 把底線換空白讓 DINO 比對友善
    names_pretty = [n.replace("_", " ") for n in names_sorted]
    return " . ".join(names_pretty)


def phrase_to_class_id(phrase: str, name2id: dict, synonyms: dict):
    """
    把 DINO 的短語（phrase）對應到你的類別 id。
    規則：phrase 轉小寫；若包含任一同義詞子字串則回傳對應 id。
    配不到則回傳 None。
    """
    p = phrase.lower().strip()
    # 先試精準名稱（處理像 'traffic cone'）
    for name, cid in name2id.items():
        if name.replace("_", " ") in p:
            return cid
    # 再試同義詞
    for base_name, keys in synonyms.items():
        for k in keys:
            if k in p:
                return name2id[base_name]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, help="nuScenes dataroot(不含 samples/)")
    ap.add_argument("--coco-json", required=True, help="nuscenes_infos_train_mono3d.coco.json 路徑")
    ap.add_argument("--gdino-config", required=True, help="GroundingDINO config，例如 GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--gdino-ckpt", required=True, help="GroundingDINO 權重 .pth")
    ap.add_argument("--out", required=True, help="輸出 pth 路徑，例如 nuscenes_dino_train_pred.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.25)
    # 若不指定，將由 categories 自動生成
    ap.add_argument("--prompt", default=None, help="自訂 prompt(以 . 分隔），預設依 categories 生成")
    args = ap.parse_args()

    # 讀 coco.json
    with open(args.coco_json, "r") as f:
        coco = json.load(f)
    images = coco["images"]     # 順序就是我們輸出 list 的順序
    categories = coco["categories"]

    # 類別表／同義詞表
    id2name, name2id, synonyms = build_class_maps_from_categories(categories)

    # 組 prompt
    prompt = args.prompt if args.prompt is not None else build_prompt_from_categories(id2name)
    print("[Prompt]", prompt)

    # 載入 GroundingDINO
    model = load_model(args.gdino_config, args.gdino_ckpt, device=args.device)

    results = []  # List[BoxList]
    for img_info in tqdm(images, desc="GroundingDINO @ nuScenes"):
        file_name = img_info["file_name"]         # 例如 samples/CAM_FRONT/xxx.jpg
        W = int(img_info["width"])
        H = int(img_info["height"])

        # 如果 file_name 是相對路徑，接上 dataroot
        img_path = file_name if os.path.isabs(file_name) else os.path.join(args.img_root, file_name)

        # 讀圖
        try:
            image_source, image_tensor = load_image(img_path)  # image_source: HxWx3 RGB (numpy), image_tensor: model input
        except Exception as e:
            warnings.warn(f"[讀圖失敗] {img_path} -> 輸出空 BoxList | {e}")
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            bl = BoxList(empty_boxes, (W, H), mode="xyxy")
            bl.add_field("scores", torch.empty((0,), dtype=torch.float32))
            bl.add_field("labels", torch.empty((0,), dtype=torch.long))
            results.append(bl)
            continue

        # 推論（boxes: normalized cxcywh；scores: 0-1；phrases: list[str]）
        boxes_cxcywh, scores, phrases = predict(
            model=model,
            image=image_tensor,
            caption=prompt,
            box_threshold=args.box_thr,
            text_threshold=args.text_thr,
            device=args.device
        )

        # 轉絕對座標 xyxy
        if boxes_cxcywh.numel() > 0:
            # 注意：以 coco.json 的 width/height 為準（避免 EXIF 或 resize 差異）
            boxes_abs = boxes_cxcywh * torch.tensor([W, H, W, H], dtype=torch.float32)
            boxes_xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")

            keep_idx = []
            out_labels = []
            for i, ph in enumerate(phrases):
                cid = phrase_to_class_id(ph, name2id=name2id, synonyms=synonyms)
                if cid is not None:
                    keep_idx.append(i)
                    out_labels.append(cid)

            if len(keep_idx) > 0:
                keep_idx = torch.tensor(keep_idx, dtype=torch.long)
                boxes_xyxy = boxes_xyxy[keep_idx]
                scr = scores[keep_idx].to(dtype=torch.float32)
                # 將類別 id +1 ⇒ 1~10，符合 PreprocessedGLIP 的 map_catid
                lbl = torch.tensor(out_labels, dtype=torch.long) + 1

                # 合理裁切到影像範圍（保守：到 max(W,H)）
                m = max(W, H)
                boxes_xyxy = boxes_xyxy.clamp(min=0, max=m)
            else:
                boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                scr = torch.empty((0,), dtype=torch.float32)
                lbl = torch.empty((0,), dtype=torch.long)
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            scr = torch.empty((0,), dtype=torch.float32)
            lbl = torch.empty((0,), dtype=torch.long)

        # 建立 BoxList（與 GLIP 輸出格式一致）
        bl = BoxList(boxes_xyxy, (W, H), mode="xyxy")
        bl.add_field("scores", scr)
        bl.add_field("labels", lbl)
        results.append(bl)

    # 儲存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(results, args.out)
    print(f"[完成] 共 {len(results)} 張影像 → {args.out}")
    

if __name__ == "__main__":
    main()
