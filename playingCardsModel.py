from ultralytics import YOLO
import os

file_name = os.path.join(os.path.dirname(__file__), 'data.yaml')
# Load the model.

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    
    # Training.
    results = model.train(
    data=file_name,
    imgsz=640,
    epochs=50,
    batch=8,
    name='yolov8n_playingcardsV2'
    )
