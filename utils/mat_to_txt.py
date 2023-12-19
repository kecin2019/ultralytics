import scipy.io as scio
import h5py
import numpy as np
import os
from PIL import Image

dataFile = "data\origin\digitStruct.mat"
mat = h5py.File(dataFile)
digitStruct = mat["digitStruct"]
name_list = digitStruct["name"]
bbox_list = digitStruct["bbox"]


# 获取图片的宽和高
def get_image_size(image_path):
    img_pillow = Image.open(image_path)
    img_width = img_pillow.width
    img_height = img_pillow.height
    img_pillow.close()
    return img_width, img_height


for i in range(len(bbox_list)):
    label_dir = os.path.join("data", "origin", "labels")
    image_dir = os.path.join("data", "origin", "image")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    image_path = os.path.join("data", "origin", "image" "{}.png".format(str(i + 1)))
    img_width, img_height = get_image_size(image_path)

    with open(
        os.path.join(label_dir, "{}.txt".format(str(i + 1))), "w", encoding="utf-8"
    ) as f:
        # 获取位置信息列表
        height_list = mat[bbox_list[i][0]]["height"]
        left_list = mat[bbox_list[i][0]]["left"]
        top_list = mat[bbox_list[i][0]]["top"]
        width_list = mat[bbox_list[i][0]]["width"]
        label_list = mat[bbox_list[i][0]]["label"]

        # 遍历每个图片中bbox的位置信息
        for j in range(len(height_list)):
            if len(height_list) == 1:
                height = height_list[0][0]
                left = left_list[0][0]
                top = top_list[0][0]
                width = width_list[0][0]
                label = int(label_list[0][0])
            else:
                height = mat[height_list[j][0]][0][0]
                left = mat[left_list[j][0]][0][0]
                top = mat[top_list[j][0]][0][0]
                width = mat[width_list[j][0]][0][0]
                label = int(mat[label_list[j][0]][0][0])

            if label == 10:
                label = 0

            x = int(left + width / 2) / img_width
            y = int(top + height / 2) / img_height
            w = width / img_width
            h = height / img_height

            f.write(
                "{0} {1} {2} {3} {4}\n".format(
                    str(label), str(x), str(y), str(w), str(h)
                )
            )
