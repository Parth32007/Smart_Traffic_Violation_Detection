from ultralytics import YOLO
import supervision as sv
import cv2

# Load model
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("videos/input/traffic_1.mp4")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
out = cv2.VideoWriter(
    "videos/output/motorcycle_tracking.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ByteTrack Tracker
tracker = sv.ByteTrack()

# Box annotator
box_annotator = sv.BoxAnnotator()

MOTORCYCLE_CLASS = 3

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

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
    import os
    os.makedirs("temp/motorcycles", exist_ok=True)

    for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):

        x1, y1, x2, y2 = map(int, bbox)

        # Keep coordinates inside image boundaries
        padding = 30

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size != 0:
            cv2.imwrite(
                f"temp/motorcycles/{tracker_id}.jpg",
                crop
            )

    labels = []

    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    ):

        labels.append(
            f"Motorcycle ID:{tracker_id}"
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
        "Vehicle Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()