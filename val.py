from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO("runs\\detect\\train4\\weights\\best.pt")  # 加载自定义模型

    # 验证模型
    metrics = model.val()  # 无需参数，数据集和设置记忆
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # 包含每个类别的map50-95列表
