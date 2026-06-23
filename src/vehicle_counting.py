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
    "videos/output/counting_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Tracker
tracker = sv.ByteTrack()

# Annotator
box_annotator = sv.BoxAnnotator()

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# Counting line
LINE_Y = height // 2

# Track counted vehicles
counted_ids = set()

# Vehicle count
vehicle_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)

    # Confidence filter
    detections = detections[detections.confidence > 0.5]

    # Vehicle filter
    mask = [
        cls in vehicle_classes
        for cls in detections.class_id
    ]

    detections = detections[mask]

    # Tracking
    detections = tracker.update_with_detections(
        detections
    )

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (0, 0, 255),
        3
    )

    labels = []

    for bbox, tracker_id in zip(
        detections.xyxy,
        detections.tracker_id
    ):

        x1, y1, x2, y2 = map(int, bbox)

        center_y = (y1 + y2) // 2

        if (
            center_y > LINE_Y and
            tracker_id not in counted_ids
        ):
            counted_ids.add(tracker_id)
            vehicle_count += 1

        labels.append(
            f"ID:{tracker_id}"
        )

    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    # Draw labels
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

    # Count display
    cv2.putText(
        annotated_frame,
        f"Total Count: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    out.write(annotated_frame)

    cv2.imshow(
        "Vehicle Counting",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()