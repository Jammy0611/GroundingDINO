import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
from torchvision.ops import box_convert
import onnx
import onnxruntime as ort

from groundingdino.util.inference import load_model, annotate
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.bert = model.bert
        self.specical_tokens = model.specical_tokens

    def forward(self, 
                input_ids: torch.Tensor, 
                token_type_ids: torch.Tensor, 
                text_self_attention_masks: torch.Tensor, 
                position_ids: torch.Tensor):
        # extract text embeddings
        tokenized_for_encoder = {}
        tokenized_for_encoder["input_ids"] = input_ids
        tokenized_for_encoder["token_type_ids"] = token_type_ids
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks.type(torch.bool)
        tokenized_for_encoder["position_ids"] = position_ids

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        return bert_output["last_hidden_state"]

class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer
        self.specical_tokens = model.specical_tokens

    def forward(self, 
                image: torch.Tensor, 
                last_hidden_state: torch.Tensor, 
                attention_mask: torch.Tensor, 
                position_ids: torch.Tensor, 
                text_self_attention_masks: torch.Tensor, 
                box_threshold: torch.Tensor, 
                text_threshold: torch.Tensor):
        outputs = self.model(image, 
                             last_hidden_state, 
                             attention_mask, 
                             position_ids, 
                             text_self_attention_masks.type(torch.bool))
        prediction_logits = outputs["pred_logits"].sigmoid().squeeze(0)
        prediction_boxes = outputs["pred_boxes"].squeeze(0)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        prediction_logits = prediction_logits[mask]
        prediction_input_ids_mask = prediction_logits > text_threshold
        prediction_boxes = prediction_boxes[mask]

        return (prediction_logits.max(dim=1)[0].unsqueeze(0), 
                prediction_boxes.unsqueeze(0), 
                prediction_input_ids_mask.unsqueeze(0))

def export_encoder(model, output):
    onnx_file = output + "/" + "gdino.encoder.onnx"
    caption = preprocess_caption("watermark")
    tokenized = model.tokenizer(caption, padding="longest", return_tensors="pt")

    (
        text_self_attention_masks, 
        position_ids
    ) = generate_masks_with_special_tokens(tokenized, model.specical_tokens, model.tokenizer)

    torch.onnx.export(
        model,
        args = (
            tokenized["input_ids"].type(torch.int).to("cpu"), 
            tokenized["token_type_ids"].type(torch.int).to("cpu"),
            text_self_attention_masks.type(torch.uint8).to("cpu"),
            position_ids.type(torch.int).to("cpu"),
        ), 
        f = onnx_file,
        input_names = [ "input_ids", "token_type_ids", "text_self_attention_masks", "position_ids" ],
        output_names = [ "last_hidden_state" ], 
        opset_version = 17, 
        export_params = True, 
        do_constant_folding = True,
        dynamic_axes = {
            "input_ids": { 1: "token_num" },
            "token_type_ids": { 1: "token_num" },
            "text_self_attention_masks": { 1: "token_num", 2: "token_num" },
            "position_ids": { 1: "token_num" },
            "last_hidden_state": { 1: "token_num" }
        },
    )

    print("export gdino.encoder.onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("check gdino.encoder.onnx ok!")

def export_decoder(model, output, encoder):
    onnx_file = output + "/" + "gdino.decoder.onnx"
    caption = preprocess_caption("watermark")

    tokenized, last_hidden_state = inference_encoder_onnx(encoder, output, caption)

    box_threshold = torch.tensor(0.35, dtype=torch.float32)
    text_threshold = torch.tensor(0.25, dtype=torch.float32)

    torch.onnx.export(
        model,
        args = (
            torch.rand(1, 3, 800, 800).type(torch.float32).to("cpu"), 
            last_hidden_state, 
            tokenized["attention_mask"].type(torch.uint8).to("cpu"), 
            tokenized["position_ids"].type(torch.int).to("cpu"), 
            tokenized["text_self_attention_masks"].type(torch.uint8).to("cpu"), 
            box_threshold, 
            text_threshold),
        f = onnx_file,
        input_names = [ "image", "last_hidden_state", "attention_mask", 
                        "position_ids", "text_self_attention_masks", 
                        "box_threshold", "text_threshold" ],
        output_names = [ "logits", "boxes", "masks" ], 
        opset_version = 17, 
        export_params = True, 
        do_constant_folding = True,
        dynamic_axes = {
            "last_hidden_state": { 1: "token_num" },
            "attention_mask": { 1: "token_num" },
            "position_ids": { 1: "token_num" },
            "text_self_attention_masks": { 1: "token_num", 2: "token_num" }
        },
    )

    print("export gdino.decoder.onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("check gdino.decoder.onnx ok!")

def inference_encoder_onnx(model, output, caption: str = None):
    onnx_file = output + "/" + "gdino.encoder.onnx"
    session = ort.InferenceSession(onnx_file)

    if caption:
        proc_caption = preprocess_caption(caption)
    else:
        proc_caption = preprocess_caption("watermark. cat. dog")
    tokenized = model.tokenizer(proc_caption, padding="longest", return_tensors="pt")

    (
        text_self_attention_masks, 
        position_ids
    ) = generate_masks_with_special_tokens(tokenized, model.specical_tokens, model.tokenizer)

    tokenized["text_self_attention_masks"] = text_self_attention_masks
    tokenized["position_ids"] = position_ids

    outputs = session.run(None, {
        "input_ids": tokenized["input_ids"].numpy().astype(np.int32), 
        "token_type_ids": tokenized["token_type_ids"].numpy().astype(np.int32),
        "text_self_attention_masks": tokenized["text_self_attention_masks"].numpy().astype(np.uint8),
        "position_ids": tokenized["position_ids"].numpy().astype(np.int32)
    })

    if caption == None:
        print(outputs)

    last_hidden_state = torch.from_numpy(outputs[0]).type(torch.float32)
    return tokenized, last_hidden_state

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    image_bgr = cv2.resize(image_bgr, (800, 800))
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

def inference_decoder_onnx(model, output):
    image = cv2.imread('asset/1.jpg')
    processed_image = preprocess_image(image).unsqueeze(0)

    caption = "watermark. glasses"

    tokenized, last_hidden_state = inference_encoder_onnx(model, output, caption)

    print(tokenized)
    print(last_hidden_state)

    onnx_file = output + "/" + "gdino.decoder.onnx"
    session = ort.InferenceSession(onnx_file)

    box_threshold = torch.tensor(0.35, dtype=torch.float32)
    text_threshold = torch.tensor(0.25, dtype=torch.float32)

    decode_outputs = session.run(None, {
        "image": processed_image.numpy().astype(np.float32),
        "last_hidden_state": last_hidden_state.numpy().astype(np.float32), 
        "attention_mask": tokenized["attention_mask"].numpy().astype(np.uint8), 
        "position_ids": tokenized["position_ids"].numpy().astype(np.int32), 
        "text_self_attention_masks": tokenized["text_self_attention_masks"].numpy().astype(np.uint8), 
        "box_threshold": box_threshold.numpy().astype(np.float32),
        "text_threshold": text_threshold.numpy().astype(np.float32) 
    })

    prediction_logits = torch.from_numpy(decode_outputs[0]) 
    prediction_boxes = torch.from_numpy(decode_outputs[1]) 
    prediction_masks = torch.from_numpy(decode_outputs[2]) 

    input_ids = tokenized["input_ids"][0].tolist()
    phrases = []
    for mask in prediction_masks[0]:
        prediction_token_ids = [input_ids[i] for i in mask.nonzero(as_tuple=True)[0].tolist()]
        phrases.append(model.tokenizer.decode(prediction_token_ids).replace('.', ''))

    with torch.no_grad():
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = annotate(image, prediction_boxes[0], prediction_logits[0], phrases)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to ONNX", add_help=True)
    parser.add_argument("--encode", "-e", help="test encoder.onnx model", action="store_true")
    parser.add_argument("--decode", "-d", help="test decoder.onnx model", action="store_true")
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    source_model = load_model(model_config_path = config_file, 
                              model_checkpoint_path = checkpoint_path, 
                              device = "cpu").to("cpu")

    encoder = Encoder(source_model)
    decoder = Decoder(source_model)

    if args.encode:
        inference_encoder_onnx(encoder, output_dir)
    elif args.decode:
        inference_decoder_onnx(decoder, output_dir)
    else:
        export_encoder(encoder, output_dir)
        export_decoder(decoder, output_dir, encoder)
