from ultralytics import YOLO
import os
import torch

file_name = os.path.join(os.path.dirname(__file__), 'data.yaml')
# Load the model.

currdevice = 'cuda' if torch.cuda.is_available() else 'cpu'
device =torch.device(currdevice)
if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    model.to(device = device)
    # Training.
    results = model.train(
    data=file_name,
    imgsz=640,
    epochs=100,
    batch=32,
    name='yolov8s_pokerChipsV1'
    )
