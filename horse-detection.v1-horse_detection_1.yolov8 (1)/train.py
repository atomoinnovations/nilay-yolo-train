from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model using your dataset
model.train(
    data=r"C:\Users\Admin\Downloads\horse-detection.v1-horse_detection_1.yolov8 (1)\data.yaml",
    epochs=10,
    imgsz=640
)

