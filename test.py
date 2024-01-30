from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO("runs/detect/train4/weights/best.pt")
    # 定义包含图像和视频文件用于推理的目录路径
    source = "datasets/tree/images/test"
    # 对来源进行推理
    results = model(source, stream=True)  # Results 对象的生成器
    # 处理结果生成器
    for result in results:
        boxes = result.boxes  # 边界框输出的 Boxes 对象
        masks = result.masks  # 分割掩码输出的 Masks 对象
        keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
        probs = result.probs  # 分类输出的 Probs 对象
