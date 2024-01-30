import random
import shutil
import os


def main():
    img_base_file = "datasets/tree/images/"
    txt_base_file = "datasets/tree/labels/"

    # 获取文件列表
    img_files = os.listdir(img_base_file)
    txt_files = os.listdir(txt_base_file)

    # 按照8:1:1的比例划分数据集
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.01

    train_len = int(len(img_files) * train_ratio)
    val_len = int(len(img_files) * val_ratio)
    test_len = int(len(img_files) * test_ratio)

    # 随机排序文件名
    img_files = random.sample(img_files, len(img_files))

    # 确保文件夹存在
    os.makedirs(img_base_file + "train/", exist_ok=True)
    os.makedirs(txt_base_file + "train/", exist_ok=True)
    os.makedirs(img_base_file + "val/", exist_ok=True)
    os.makedirs(txt_base_file + "val/", exist_ok=True)
    os.makedirs(img_base_file + "test/", exist_ok=True)
    os.makedirs(txt_base_file + "test/", exist_ok=True)

    # 复制文件到相应的目录
    for i, file in enumerate(img_files):
        if i < train_len:
            shutil.move(
                os.path.join(img_base_file, file),
                os.path.join(img_base_file, "train/", file),
            )
            shutil.move(
                os.path.join(txt_base_file, file.replace(".JPG", ".txt")),
                os.path.join(txt_base_file, "train/", file.replace(".JPG", ".txt")),
            )
        elif i < train_len + val_len:
            shutil.move(
                os.path.join(img_base_file, file),
                os.path.join(img_base_file, "val/", file),
            )
            shutil.move(
                os.path.join(txt_base_file, file.replace(".JPG", ".txt")),
                os.path.join(txt_base_file, "val/", file.replace(".JPG", ".txt")),
            )
        else:
            shutil.move(
                os.path.join(img_base_file, file),
                os.path.join(img_base_file, "test/", file),
            )
            shutil.move(
                os.path.join(txt_base_file, file.replace(".JPG", ".txt")),
                os.path.join(txt_base_file, "test/", file.replace(".JPG", ".txt")),
            )


if __name__ == "__main__":
    main()
