import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, filedialog
from PIL import Image, ImageTk


import cv2
import numpy as np

def img_read(path):
    """
    读取图片并调整大小以及转为灰度图
    """
    img = cv2.imread(path)  # 读取图片
    img = cv2.resize(img, (200, 100))  # 调整图片大小
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    return img, gray

def Gaus(gray):
    """
    对灰度图进行高斯滤波
    """
    gaus = cv2.GaussianBlur(gray, (7, 7), 0)  # 高斯滤波
    return gaus

def sobel(gaus):
    """
    对高斯滤波后的图进行Sobel边缘检测
    """
    grad_x = cv2.Sobel(gaus, ddepth=cv2.CV_32F, dx=1, dy=0)  # Sobel x方向
    grad_y = cv2.Sobel(gaus, ddepth=cv2.CV_32F, dx=0, dy=1)  # Sobel y方向
    gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))  # 计算梯度
    return grad_x, grad_y, gradient

def apaptivTresh(gradient):
    """
    对梯度图进行自适应阈值处理
    """
    blur = Gaus(gradient)  # 再次进行高斯滤波
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 10
    )  # 自适应阈值处理
    return thresh

def point_out(morph):
    """
    寻找轮廓并获取矩形顶点
    """
    (cnts, _) = cv2.findContours(morph.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    c = sorted(cnts, key=cv2.contourArea, reverse=False)[0]  # 按面积排序并取最大的轮廓
    rect = cv2.minAreaRect(c)  # 获取最小矩形
    rect = cv2.boxPoints(rect)  # 获取矩形的顶点
    box = np.intp(rect)  # 转换为整数点
    return box

def draw_cut(img, box):
    """
    绘制矩形并裁剪图片
    """
    drawing = cv2.drawContours(img.copy(), [box], -1, (0, 255, 0), 2)  # 绘制矩形
    Xs = [i[0] for i in box]  # 获取X坐标
    Ys = [i[1] for i in box]  # 获取Y坐标
    x1 = min(Xs)  # 获取X坐标的最小值
    x2 = max(Xs)  # 获取X坐标的最大值
    y1 = min(Ys)  # 获取Y坐标的最小值
    y2 = max(Ys)  # 获取Y坐标的最大值
    hight = y2 - y1  # 计算高度
    width = x2 - x1  # 计算宽度
    crop_img = img[y1 : y1 + hight, x1 : x1 + width]  # 裁剪图片
    return drawing, crop_img


class MorphologyApp:
    def __init__(self, root, img_path):
        # 初始化窗口和图像
        self.root = root
        self.img, self.gray_img = img_read(img_path)
        self.img_blur = Gaus(self.gray_img)
        self.gradX, self.gradY, self.gradient = sobel(self.img_blur)
        self.thresh = apaptivTresh(self.gradient)

        # 创建两个画布，一个用于显示处理后的图像，一个用于显示轮廓提取结果
        self.canvas_processed = tk.Canvas(
            root, width=self.thresh.shape[1], height=self.thresh.shape[0]
        )
        self.canvas_processed.pack()
        self.canvas_contour = tk.Canvas(
            root, width=self.thresh.shape[1], height=self.thresh.shape[0]
        )
        self.canvas_contour.pack()

        # 初始化内核大小
        self.kernel_sizes = {
            "闭运算": (42, 10),
            "横向": (30, 10),
            "纵向": (35, 43),
        }

        # 创建滑块用于调整内核大小
        self.sliders = {}
        for morph_type, initial_size in self.kernel_sizes.items():
            label = tk.Label(root, text=f"{morph_type} 内核大小")
            label.pack()
            slider_x = Scale(
                root,
                from_=1,
                to=50,
                orient="horizontal",
                length=200,
                resolution=1,
                command=self.update_morphology,
            )
            slider_y = Scale(
                root,
                from_=1,
                to=50,
                orient="horizontal",
                length=200,
                resolution=1,
                command=self.update_morphology,
            )
            slider_x.set(initial_size[0])
            slider_y.set(initial_size[1])
            slider_x.pack()
            slider_y.pack()
            self.sliders[morph_type] = (slider_x, slider_y)

        # 从滑块获取当前内核大小
        self.current_kernel_sizes = {
            morph_type: (
                self.sliders[morph_type][0].get(),
                self.sliders[morph_type][1].get(),
            )
            for morph_type in self.kernel_sizes
        }

        # 应用形态学操作
        self.result = self.img_morphology(self.thresh, self.current_kernel_sizes)

        # 显示原始图像
        self.update_canvas()

    def update_morphology(self, *_):
        # 使用当前的内核大小更新形态学操作的结果
        self.result = self.img_morphology(self.thresh, self.current_kernel_sizes)
        
        # 更新当前内核大小的字典
        self.current_kernel_sizes = {
            morph_type: (
                self.sliders[morph_type][0].get(),
                self.sliders[morph_type][1].get(),
            )
            for morph_type in self.kernel_sizes
        }

        # 显示形态学操作处理后的图像
        self.update_canvas(self.result)

    def save_image(self):
        # 打开文件对话框供用户选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")]
        )

        # 如果用户取消保存操作，则直接返回
        if not file_path:
            return

        # 保存处理后的图像到指定路径
        cv2.imwrite(file_path, self.result)

    def img_morphology(self, image, kernel_sizes, iterations=1):
        # 使用形态学操作处理图像
        # 获取闭运算所需要的核大小
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_sizes["闭运算"])
        # 对图像进行闭运算
        close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)

        # 获取横向操作所需要的核大小
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_sizes["横向"])
        # 对闭运算后的图像进行侵蚀操作
        erode_x = cv2.morphologyEx(close, cv2.MORPH_ERODE, kernel_x, iterations=iterations)
        # 对侵蚀后的图像进行膨胀操作
        dilate_x = cv2.morphologyEx(erode_x, cv2.MORPH_DILATE, kernel_x, iterations=iterations)

        # 获取纵向操作所需要的核大小
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_sizes["纵向"])
        # 对纵向进行侵蚀操作
        erode_y = cv2.morphologyEx(dilate_x, cv2.MORPH_ERODE, kernel_y, iterations=iterations)
        # 对纵向进行膨胀操作
        dilate_y = cv2.morphologyEx(erode_y, cv2.MORPH_DILATE, kernel_y, iterations=iterations)

        # 返回纵向膨胀后的图像
        return dilate_y

    def update_canvas(self, image=None):
        if image is None:
            image = self.result

        # 在一个画布上显示处理后的图像
        image_tk = self.convert_cv_to_tk(image)
        self.canvas_processed.create_image(0, 0, anchor="nw", image=image_tk)

        # 提取轮廓并在原始图像上绘制
        box = point_out(image)
        draw_img, crop_img = draw_cut(self.img.copy(), box)

        # 转换并在另一个画布上显示带有轮廓的图像
        draw_img_tk = self.convert_cv_to_tk(draw_img)
        self.canvas_contour.create_image(0, 0, anchor="nw", image=draw_img_tk)

        # 存储对PhotoImage对象的引用
        self.image_tk_processed = image_tk
        self.image_tk_contour = draw_img_tk

        self.root.update_idletasks()

    def convert_cv_to_tk(self, cv_image):
        # 将OpenCV图像转换为RGB格式
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 将RGB图像转换为PhotoImage
        image_tk = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image=image_tk)

        return image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = MorphologyApp(root, "data\\origin\\1000.png")
    root.mainloop()
