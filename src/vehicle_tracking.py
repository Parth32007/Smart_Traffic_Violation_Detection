from ultralytics import YOLO
import supervision as sv
import cv2

# Load model
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("videos/input/traffic.mp4")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
out = cv2.VideoWriter(
    "videos/output/tracked_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ByteTrack Tracker
tracker = sv.ByteTrack()

# Box annotator
box_annotator = sv.BoxAnnotator()

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.5]

    detections = detections[
        detections.class_id != None
    ]

    # Filter vehicle classes
    mask = [
        class_id in vehicle_classes
        for class_id in detections.class_id
    ]

    detections = detections[mask]

    # Tracking
    detections = tracker.update_with_detections(
        detections
    )

    labels = []

    for class_id, tracker_id in zip(
        detections.class_id,
        detections.tracker_id
    ):

        class_name = model.names[class_id]

        labels.append(
            f"ID:{tracker_id}"
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