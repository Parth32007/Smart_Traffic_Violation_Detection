from ultralytics import YOLO
import supervision as sv
import cv2
import os

# Load model
vehicle_model = YOLO("yolov8s.pt")

helmet_model = YOLO(
    "models/helmet_detector/best.pt"
)

# Open video
cap = cv2.VideoCapture("videos/input/traffic_1.mp4")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
out = cv2.VideoWriter(
    "videos/output/helmet_violation_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ByteTrack Tracker
tracker = sv.ByteTrack()

# Box annotator
box_annotator = sv.BoxAnnotator()

MOTORCYCLE_CLASS = 3

os.makedirs("temp/motorcycles", exist_ok=True)

while True:

    helmet_predictions = {}

    ret, frame = cap.read()

    if not ret:
        break

    results = vehicle_model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = detections[
        detections.class_id != None
    ]

    # Filter vehicle classes
    mask = detections.class_id == MOTORCYCLE_CLASS
    detections = detections[mask]

    # Tracking
    detections = tracker.update_with_detections(
        detections
    )

    # Create folder to save motorcycle crops
    for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):

        x1, y1, x2, y2 = map(int, bbox)

        # Keep coordinates inside image boundaries
        padding = 60

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        helmet_results = helmet_model.predict(
            source=crop,
            conf=0.15,
            verbose=False
        )

        helmet_label = "Unknown"

        if len(helmet_results[0].boxes) > 0:

            cls = int(helmet_results[0].boxes[0].cls[0])

            helmet_label = helmet_model.names[cls]

        helmet_predictions[tracker_id] = helmet_label

        print(f"Motorcycle ID:{tracker_id} -> {helmet_label}")

    labels = []

    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    ):

        labels.append(
            f"ID:{tracker_id} | {helmet_predictions[tracker_id]}"
        )

    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    for bbox, label in zip(
        detections.xyxy,
        labels
    ):

        x1, y1, x2, y2 = map(int, bbox)

        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(annotated_frame)

    cv2.imshow(
        "Helmet Violation Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()