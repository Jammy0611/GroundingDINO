import numpy as np
import cv2
import supervision as sv

from groundingdino.util.inference import Model, annotate

image = cv2.imread("asset/cat_dog.jpeg")
caption = "cat . dog"

model = Model("groundingdino/config/GroundingDINO_SwinT_OGC.py", 
              "weights/groundingdino_swint_ogc.pth", 
              "cpu")

detections, phrases = model.predict_with_caption(image, caption)

labels = [ f"{phrase}" for phrase in phrases ]

bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
annotated_frame = bbox_annotator.annotate(scene=image, detections=detections)
annotated_frame = label_annotator.annotate(scene=image, detections=detections, labels=labels)

cv2.imshow("image", annotated_frame)
cv2.waitKey()
cv2.destroyAllWindows()
