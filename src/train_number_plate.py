from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")

    # Train the model
    model.train(
        data="datasets/number_plate_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        name="number_plate_detector"
    )

if __name__ == "__main__":
    main()