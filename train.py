from ultralytics import YOLO


if __name__ == "__main__":
    # 加载模型文件
    model = YOLO("yolov8n.yaml")

    # 训练模型
    results = model.train(
        data="housenumber.yaml",
        epochs=100,
        imgsz=640,
        workers=4,
        save_period=1,
        amp=False,
        batch=16,
    )
