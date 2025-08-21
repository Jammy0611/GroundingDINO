from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "./weights/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915246912465.jpg"
# TEXT_PROMPT = "car . vehicle . sedan . van . taxi . pickup . house"
TEXT_PROMPT = "car . vehicle . house"
BOX_TRESHOLD = 0.23
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("./image/annotated_image.jpg", annotated_frame)