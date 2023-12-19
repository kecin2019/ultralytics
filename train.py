from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8m.yaml").load(
        "model\yolov8m.pt"
    )  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="scripts\housenumber.yaml", epochs=100, imgsz=640)
