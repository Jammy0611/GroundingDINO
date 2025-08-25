"""
Visualize nuScenes Ground Truth and DINO Predictions
--------------------------------------------------

This script overlays both ground truth bounding boxes and predicted bounding
boxes from a trained Grounding DINO model onto nuScenes camera images.

Input files
~~~~~~~~~~~~

* ``nuscenes_dino_train_pred.pth`` – a PyTorch checkpoint containing a list of
  ``BoxList`` objects. Each ``BoxList`` corresponds to one image in the nuScenes
  training set. The bounding boxes are stored in ``xyxy`` format and the
  associated labels and scores are found in the ``extra_fields`` dictionary of
  each ``BoxList``.
* ``nuscenes_infos_train_mono3d.coco.json`` – a COCO‑style annotation file for
  the nuScenes training set containing metadata for every image and its
  ground truth bounding boxes. Bounding boxes are stored in ``[x, y, w, h]``
  format and need to be converted to ``[x1, y1, x2, y2]`` for drawing.

The script aligns predictions to images by **index**: the first element in the
prediction list corresponds to the first entry in the ``images`` array within
the COCO JSON. This ordering is consistent when the predictions were
generated sequentially over the dataset. If your predictions are not ordered
this way, consider building a mapping from image tokens to prediction indices
instead.

Usage
~~~~~

Run the script with Python. By default it assumes the nuScenes image files
live under ``/data/sets/nuscenes``. Adjust ``DATA_ROOT`` below to point to
your own nuScenes data directory if necessary. The rendered images are
written into a subdirectory named ``image`` relative to the script’s location.

::

    python visualize_nuscenes_predictions.py \
        --predictions nuscenes_dino_train_pred.pth \
        --annotations nuscenes_infos_train_mono3d.coco.json \
        --data_root /path/to/nuscenes \
        --output_dir image

Notes
~~~~~

* This script draws ground truth boxes in **green** and predicted boxes in
  **red**. It also annotates each box with its category name. Feel free to
  adjust colors or omit text labels if the display becomes cluttered.
* Processing the entire nuScenes training set can take significant time and
  disk space (~168k images). To visualise a subset, you may pass the
  ``--max_images`` flag to limit the number of frames processed.

"""

import argparse
import json
import os
from collections import defaultdict
from typing import List, Tuple, Dict

try:
    import torch  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required to load the prediction file. Please install torch "
        "or run this script in an environment where torch is available."
    ) from e

from PIL import Image, ImageDraw, ImageFont  # type: ignore

def _measure_text(draw, font, text: str):
    # Pillow ≥ 10：首選 textbbox
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Pillow ≥ 8：font.getbbox
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    # 最後退路（較舊版 Pillow）
    return font.getsize(text)

def load_annotations(ann_path: str) -> Tuple[List[dict], Dict[str, List[Tuple[float, float, float, float, str]]], Dict[int, str]]:
    """Load the COCO JSON file and prepare ground truth structures.

    Parameters
    ----------
    ann_path: str
        Path to the COCO‑formatted annotations JSON file.

    Returns
    -------
    images: list of dict
        List of image metadata dictionaries from the JSON.
    gt_map: dict
        Dictionary mapping image IDs to lists of ground truth boxes and
        category names in ``(x1, y1, x2, y2, category_name)`` format.
    catid2name: dict
        Dictionary mapping category IDs to their human readable names.
    """
    with open(ann_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    # Build a lookup from category id to category name
    catid2name: Dict[int, str] = {cat['id']: cat['name'] for cat in categories}

    # Build ground truth box map: image id -> list of (x1,y1,x2,y2,cat_name)
    gt_map: Dict[str, List[Tuple[float, float, float, float, str]]] = defaultdict(list)
    for ann in annotations:
        # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        cat_id = ann['category_id']
        cat_name = catid2name.get(cat_id, str(cat_id))
        image_id = ann['image_id']
        gt_map[image_id].append((x1, y1, x2, y2, cat_name))

    return images, gt_map, catid2name


def draw_boxes(
    image: Image.Image,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[str],
    color: Tuple[int, int, int],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    alpha: float = 1.0,
) -> None:
    """Draw bounding boxes with labels on an image.

    Parameters
    ----------
    image: PIL.Image.Image
        The PIL image to draw on. Not modified here; drawing is performed via ``draw``.
    boxes: list of tuples
        Bounding boxes in [x1, y1, x2, y2] format.
    labels: list of str
        Category names associated with each bounding box. Must be the same length as ``boxes``.
    color: tuple
        RGB color for the bounding box lines and text.
    draw: PIL.ImageDraw.ImageDraw
        A drawing context obtained from the image.
    font: PIL.ImageFont.FreeTypeFont
        Font used for label text.
    alpha: float
        Transparency for the rectangle fill. Not currently used, but reserved for future use.
    """
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        # Draw rectangle outline
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        # Draw label background for readability
        text = label
        if text:
            tw, th = _measure_text(draw, font, text)
            # 避免標籤被畫到影像外，往上貼齊 0
            y_text = max(0, y1 - th)
            text_bg = (x1, y_text, x1 + tw + 4, y_text + th)
            draw.rectangle(text_bg, fill=(255, 255, 255))
            draw.text((x1 + 2, y_text), text, fill=color, font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay ground truth and prediction bounding boxes on nuScenes images.")
    parser.add_argument('--predictions', type=str, required=True, help='Path to the .pth file containing DINO predictions')
    parser.add_argument('--annotations', type=str, required=True, help='Path to the COCO‑style JSON file with annotations')
    parser.add_argument('--data_root', type=str, default='/data/sets/nuscenes', help='Root directory for nuScenes images')
    parser.add_argument('--output_dir', type=str, default='image', help='Directory to save visualized images')
    parser.add_argument('--max_images', type=int, default=None, help='Process at most this many images (for testing)')
    args = parser.parse_args()

    pred_path = args.predictions
    ann_path = args.annotations
    data_root = args.data_root
    out_dir = args.output_dir
    max_images = args.max_images

    # Load predictions
    print(f"Loading predictions from {pred_path} ...")
    predictions = torch.load(pred_path, map_location='cpu')
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a list of BoxList objects in {pred_path}, got {type(predictions)}")

    # Load annotations and ground truth boxes
    print(f"Loading annotations from {ann_path} ...")
    images, gt_map, catid2name = load_annotations(ann_path)

    if len(predictions) != len(images):
        print(
            f"Warning: number of prediction entries ({len(predictions)}) does not match number of images "
            f"({len(images)}). Proceeding by pairing predictions to images by index."
        )

    # Prepare the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Preload a font. Use a default PIL font if no TrueType font is available.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()

    # Default colours: ground truth (green) and predictions (red)
    GT_COLOR = (0, 255, 0)
    PRED_COLOR = (255, 0, 0)

    total_images = len(images) if max_images is None else min(len(images), max_images)
    for idx, image_info in enumerate(images[:total_images]):
        # Retrieve the prediction for this index, if available
        try:
            pred = predictions[idx]
        except IndexError:
            pred = None

        # Get ground truth list for this image
        image_id = image_info['id']
        gt_entries = gt_map.get(image_id, [])

        # Compose the image path
        file_name = image_info['file_name']
        image_path = os.path.join(data_root, file_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found; skipping.")
            continue

        # Open the image
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to open image {image_path}: {e}")
            continue

        draw = ImageDraw.Draw(img)

        # Draw ground truth boxes
        if gt_entries:
            gt_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in gt_entries]
            gt_labels = [label for (_, _, _, _, label) in gt_entries]
            draw_boxes(img, gt_boxes, gt_labels, GT_COLOR, draw, font)

        # Draw prediction boxes
        if pred is not None:
            # Each ``BoxList`` exposes .bbox and .extra_fields with 'labels'
            try:
                pred_boxes = pred.bbox.tolist()
                pred_labels = pred.extra_fields.get('labels')
                # Convert to python list
                if hasattr(pred_labels, 'tolist'):
                    pred_labels = pred_labels.tolist()
                # Map predicted label ids to category names if possible; otherwise use id
                pred_names = [catid2name.get(lbl, str(lbl)) for lbl in pred_labels]
            except Exception as e:
                print(f"Failed to extract predictions at index {idx}: {e}")
                pred_boxes, pred_names = [], []

            if pred_boxes:
                draw_boxes(img, pred_boxes, pred_names, PRED_COLOR, draw, font)

        # Create a flattened filename for saving to avoid directory creation
        safe_name = file_name.replace('/', '__')
        out_path = os.path.join(out_dir, safe_name)
        try:
            img.save(out_path)
        except Exception as e:
            print(f"Failed to save visualisation to {out_path}: {e}")

        if idx % 100 == 0:
            print(f"Processed {idx + 1}/{total_images} images ...")

    print(f"Completed visualising {total_images} images. Results saved to '{out_dir}'.")


if __name__ == '__main__':
    main()