# -*- coding: utf-8 -*-
"""
@Time: 2020/7/18 10:29
@Auth: AI.JQ
@File: xml_json_to_txt.py
@IDE: PyCharm
"""

import os
import numpy as np
from xml.etree import ElementTree as ET
import cv2
import math


def box_to_center_relative(box, img_height, img_width):
    """
    Convert VOC annotations box with format [xmin, ymin, xmax, ymax] to
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    xmin, ymin, xmax, ymax = box

    xmin = max(xmin, 0)
    xmax = min(xmax - 1, img_width - 1)
    ymin = max(ymin, 0)
    ymax = min(ymax - 1, img_height - 1)

    x = (xmin + xmax) / 2 / img_width
    y = (ymin + ymax) / 2 / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    return np.array([x, y, w, h])


# xml生成class.names 存储标签类别
def class_names(anno_root, names_save_root):
    anno_files = os.listdir(anno_root)
    anno_files.sort()
    for anno_file in anno_files:
        filetype = anno_file.split(os.path.sep)[-1].split(".")[-1]  # xml json
        # print(filetype, type(filetype))  #获取文件类型 xml or json
        if filetype == "xml":
            anno_path = os.path.join(anno_root, anno_file)
            anno_path = anno_path.replace("\\", "/")  # windows 下替换\   ..../0001.xml
            tree = ET.parse(open(anno_path, "rb"))
            root = tree.getroot()
            # print(root)
            try:
                line = anno_path.split(os.path.sep)[-1].split("/")[-1]
            except:
                print("Can not get xml files!! Please check it")
                exit(0)
            for obj in root.iter("object"):
                # cls = obj.find('name').text
                cls = "tree"
                if cls not in anno:
                    anno[cls] = anno.get(cls, 0) + len(anno)

        elif filetype == "json":
            pass
        else:
            print("Just xml or json files")

    with open(names_save_root + "/class.names", "w") as f:
        for k in anno:
            f.write(str(k))
            f.write("\n")

    return anno


# xml格式转化为txt 存储在labels下
def voc_xml_to_txt(img_root, anno_root, anno_names, box_to_center=False):
    img_files = os.listdir(img_root)
    img_files.sort()
    anno_files = os.listdir(anno_root)
    anno_files.sort()

    # 在./data文件夹下创建labels文件夹存储标签
    p, _ = os.path.split(anno_root)
    dirs = p + "/labels"
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for img_file, anno_file in zip(img_files, anno_files):
        # if box_to_center: #判断是否转化为 [x_center, y_center, w, h] 并进行归一化
        #     img_path = os.path.join(img_root, img_file)
        #     img_path = img_path.replace('\\', '/')  # windows 下替换\
        #     img = cv2.imread(img_path)
        #     im_height, im_width, channels = img.shape  # 获取 W H

        anno_path = os.path.join(anno_root, anno_file)
        anno_path = anno_path.replace("\\", "/")  # windows 下替换\
        tree = ET.parse(open(anno_path, "rb"))
        root = tree.getroot()
        try:
            line = anno_path.split(os.path.sep)[-1].split("/")[-1]
        except:
            print("Can not get xml files!! Please check it")
            exit(0)

        bbox_labels = []
        im_height = int(tree.findtext("./size/height"))
        im_width = int(tree.findtext("./size/width"))
        for obj in root.iter("object"):
            bbox_sample = []
            # cls = obj.find('name').text  # 获得类别名称
            cls = "tree"
            bbox = obj.find("bndbox")
            bbox = [
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            ]
            if math.isnan(bbox[0]):
                continue

            if box_to_center:  # 是否转化为[x_center, y_center, w, h] 并进行归一化
                bbox = box_to_center_relative(bbox, im_height, im_width)

            bbox_sample.append(int(anno_names[cls]))
            bbox_sample.append(float(bbox[0]))
            bbox_sample.append(float(bbox[1]))
            bbox_sample.append(float(bbox[2]))
            bbox_sample.append(float(bbox[3]))
            bbox_labels.append(bbox_sample)

        anno_txt = anno_file.replace(".xml", "")
        with open((dirs + "/" + anno_txt + ".txt"), "w") as f:
            for j in bbox_labels:
                for k in j:
                    f.write(str(k) + " ")
                f.write("\n")


def coco_json_to_txt(images_root, anno_root, anno_names, box_to_center=False):
    pass


def creat_set(img_path_list, train=0.7, test=0.3, val=0):
    """
    img_path_list = []
    datasets_path = os.listdir(datasets_root) # [data1,data2....]

    for path in datasets_path:
        data_root = os.path.join(datasets_root, path)
        data_root = data_root.replace('\\', '/')  # windows 下替换\    ./datasets/datasets/data1
        images_root = data_root + './images'      # 存储图像文件夹的路径  ./datasets/datasets/data1/images
        for img_file in os.listdir(images_root):
            img_path = os.path.join(datasets_root, img_file)
            img_path = img_path.replace('\\', '/')  # windows 下替换\
            img_path_list.append(img_path)
    """

    # 制作测试集
    import random

    random.seed(10010)
    random.shuffle(img_path_list)
    total = len(img_path_list)

    # txt_save_path, _ = os.path.split(datasets_root)
    txt_save_path = datasets_root

    # 写入到train.txt
    train_txt = open(txt_save_path + "/train.txt", "w")
    train_files = img_path_list[: int(total * float(train))]
    for r in train_files:
        train_txt.write(r + "\t" + "\n")
    print("[INFO] Writing train.txt Finishing!!")

    # 写入到test.txt
    test_txt = open(txt_save_path + "/test.txt", "w")
    test_files = img_path_list[
        int(total * float(train)) : int(total * float(test) + total * float(train))
    ]
    for r in test_files:
        test_txt.write(r + "\t" + "\n")
    print("[INFO] Writing test.txt Finishing!!")

    # 写入到val.txt
    val_txt = open(txt_save_path + "/valid.txt", "w")
    val_files = img_path_list[
        int(total * float(test) + total * float(train)) : int(
            total * float(val) + total * float(test) + total * float(train)
        )
    ]
    for r in val_files:
        val_txt.write(r + "\t" + "\n")
    print("[INFO] Writing val.txt Finishing!!")


if __name__ == "__main__":
    datasets_root = "tree"  # 1.设置数据集文件夹路径
    # datasets_root = os.path.abspath(datasets_root)
    img_name = "/images"  # 2.设置存储图片文件夹的名字
    label_name = "/annotation"  # 3.设置存储xml标签文件夹的名字
    box_to_center_wh = True  # 4.是否将bbox转化为[xcenter,ycenter,w,h]
    train, test, val = 0.8, 0.1, 0.1  # 5. 设置切分train test val数据比例,总和为1

    names_save_root, _ = os.path.split(datasets_root)
    datasets_path = os.listdir(datasets_root)
    print(datasets_path)
    anno = {}
    img_list = []
    for path in datasets_path:
        data_root = os.path.join(datasets_root, path)
        data_root = data_root.replace(
            "\\", "/"
        )  # windows 下替换\    ./datasets/datasets/data1
        images_root = (
            data_root + img_name
        )  # 存储图像文件夹的路径  ./datasets/datasets/data1/images
        anno_root = data_root + label_name  # 存储标签文件夹的路径  ./datasets/datasets/data1/anno

        # 生成class.names文件
        # anno_names = class_names(anno_root, names_save_root)
        anno_names = class_names(anno_root, datasets_root)
        # xml or json数据格式转为txt，并存储于labels
        anno_files = os.listdir(anno_root)
        filetype = anno_files[0].split(os.path.sep)[-1].split(".")[-1]
        # print(filetype, type(filetype))  #获取文件类型 xml or json
        if filetype == "xml":
            voc_xml_to_txt(
                images_root, anno_root, anno_names, box_to_center=box_to_center_wh
            )
        elif filetype == "json":
            coco_json_to_txt(
                images_root, anno_root, anno_names, box_to_center=box_to_center_wh
            )
        else:
            print("Only xml or json files")

        # 将所有图片路径保存在img_list列表中，用以切分训练集测试集
        for img_file in os.listdir(images_root):
            img_path = os.path.join(images_root, img_file)
            img_path = img_path.replace(
                "\\", "/"
            )  # windows 下替换\  ./datasets/datasets/data1/images/xxx.jpg
            img_list.append(img_path)

    # 数据切分 train/test/val, 切分比例自定义
    creat_set(img_list, train=train, test=test, val=val)

    # 生成train.data文件
    # classes=3
    # train=./data/train.txt
    # test=./data/test.txt
    # valid=./data/valid.txt
    # names=./data/class.names
    # dirs, _ = os.path.split(datasets_root)
    dirs = datasets_root
    with open((datasets_root + "/train.data"), "w") as f:
        f.write("classes=" + str(len(anno_names)))
        f.write("\n")
        f.write("train=" + (dirs + "/train.txt"))
        f.write("\n")
        f.write("test=" + (dirs + "/test.txt"))
        f.write("\n")
        f.write("valid=" + (dirs + "/valid.txt"))
        f.write("\n")
        f.write("names=" + (dirs + "/class.names"))
