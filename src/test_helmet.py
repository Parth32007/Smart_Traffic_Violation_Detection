from ultralytics import YOLO
import cv2
import os

# Load custom trained model
model = YOLO("models/helmet_detector/best.pt")

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "test_results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Class names
class_names = model.names

for image_name in os.listdir(INPUT_FOLDER):

    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(INPUT_FOLDER, image_name)

        results = model.predict(
            source=image_path,
            conf=0.5,
            save=False,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        output_path = os.path.join(OUTPUT_FOLDER, image_name)
        cv2.imwrite(output_path, annotated)

        helmet_count = 0
        no_helmet_count = 0

        for box in result.boxes:
            cls = int(box.cls[0])

            if class_names[cls] == "With Helmet":
                helmet_count += 1
            elif class_names[cls] == "Without Helmet":
                no_helmet_count += 1

        print(f"\nImage: {image_name}")
        print(f"With Helmet     : {helmet_count}")
        print(f"Without Helmet  : {no_helmet_count}")
        print("-" * 40)

print("\n✅ Testing Completed!")
print("Annotated images saved in: test_results/")