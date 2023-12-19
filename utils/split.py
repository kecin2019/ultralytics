import random
import shutil


def main():
    random.seed(111)

    # 定义总范围
    total_range = [i for i in range(1, 33403)]

    # 随机打乱总范围
    random.shuffle(total_range)

    # 定义各部分的长度
    length1 = 10000
    length2 = 5000
    length3 = 18402

    # 分割数字范围
    train = total_range[:length1]
    val = total_range[length1 : length1 + length2]
    test = total_range[length1 + length2 : length1 + length2 + length3]

    img_base_file = (
        "D:/Project/pythonProject/House_number_detection/data/origin/images/"
    )
    txt_base_file = (
        "D:/Project/pythonProject/House_number_detection/data/origin/labels/"
    )

    for i in train:
        shutil.move(
            img_base_file + str(i) + ".png", img_base_file + "train/" + str(i) + ".png"
        )
        shutil.move(
            txt_base_file + str(i) + ".txt", txt_base_file + "train/" + str(i) + ".txt"
        )

    for j in val:
        shutil.move(
            img_base_file + str(j) + ".png", img_base_file + "val/" + str(j) + ".png"
        )
        shutil.move(
            txt_base_file + str(j) + ".txt", txt_base_file + "val/" + str(j) + ".txt"
        )

    for k in test:
        shutil.move(
            img_base_file + str(k) + ".png", img_base_file + "test/" + str(k) + ".png"
        )
        shutil.move(
            txt_base_file + str(k) + ".txt", txt_base_file + "test/" + str(k) + ".txt"
        )


if __name__ == "__main__":
    main()
