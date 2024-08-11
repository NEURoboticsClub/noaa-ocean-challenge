from ultralytics import YOLO
import json
import os

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="sterfesh.yaml",epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set

results = model.predict(source="seafloor_footage.mp4",save=True)