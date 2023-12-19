from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    model = YOLO("runs\\detect\\train7\\weights\\best.pt")
    # 定义包含图像和视频文件用于推理的目录路径
    source = "datasets\\housenumber\\images\\test"
    # 对来源进行推理
    results = model(source, stream=True)  # Results 对象的生成器
