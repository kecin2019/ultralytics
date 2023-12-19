from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("model\yolov8m.pt")

    # Train the model
    results = model.train(
        data="scripts\housenumber.yaml",
        epochs=100,
        imgsz=640,
        workers=4,
        save_period=10,
    )
