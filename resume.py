from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO("runs\\detect\\train7\\weights\\best.pt")

    # 恢复训练
    results = model.train(resume=True)
