from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="datasets/helmet_dataset_v2/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=0,
        name="helmet_detector_v2"
    )

if __name__ == "__main__":
    main()