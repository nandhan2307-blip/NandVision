from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_objects(image_path):
    results = model(image_path)

    # Draw bounding boxes and labels
    annotated_image = results[0].plot()

    return annotated_image


if __name__ == "__main__":
    image_path = "assets/test.jpg"

    annotated_image = detect_objects(image_path)

    # Save the detected image
    output_path = "assets/detected.jpg"

    import cv2
    cv2.imwrite(output_path, annotated_image)

    print(f"\nDetected image saved to: {output_path}")