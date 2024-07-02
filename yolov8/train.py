#!/usr/bin/env python
from ultralytics import YOLO

# Load a model
model = YOLO(r"./yolov8n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Train the model
results = model.train(data=r"../car_dataset/car.yaml", epochs=100, imgsz=1080, batch=16)

# Customize validation settings
validation_results = model.val(data=r"../car_dataset/car.yaml", imgsz=1080, batch=16, conf=0.25, iou=0.6, device="0")

# Export the model
model.export()
# model.save('trained_model_save.pt')
