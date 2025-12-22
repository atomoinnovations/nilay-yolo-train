from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Run prediction on test images
results = model.predict(
    source=0,
    conf=0.5,
    show=True
)

print("Prediction completed successfully")
