from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8m.yaml")
    model.info()  # display model information

    # Train the model
    results = model.train(
        data="housenumber.yaml",
        epochs=100,
        imgsz=640,
        workers=4,
        save_period=10,
        amp=False,
    )
