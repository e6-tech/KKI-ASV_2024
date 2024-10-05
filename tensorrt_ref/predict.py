import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.engine")
img = cv2.imread("path/to/image.jpg")

for _ in range(100):
    result = model.predict(
        [img] * 8,  # batch=8 of the same image
        verbose=False,
        device="cuda",
    )
