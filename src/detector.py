
import cv2


def detect_objects(image_path):
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    image = cv2.imread(image_path)

    results = model(image)
    annotated_image = results[0].plot()

    return annotated_image