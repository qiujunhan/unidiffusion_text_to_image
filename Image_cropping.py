import cv2
import numpy as np
import os
from glob import glob
# 加载人脸检测器模型
prototxt_path = "./deploy.prototxt.txt"
model_path = "./res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 加载输入图片并获取尺寸
image = cv2.imread("input.jpeg")
h, w = image.shape[:2]

# 进行人脸检测
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# 创建一个白色的512x512图片
output_image = np.ones((512, 512, 3), np.uint8) * 255

def adjusted_box(box):
    # 提取坐标值
    x1, y1, x2, y2 = box

    # 计算矩形的宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 根据最长边调整为正方形
    max_side = max(width, height)
    delta_width = (max_side - width) / 2
    delta_height = (max_side - height) / 2

    # 调整后的坐标值
    x1 = int(x1 - delta_width)
    y1 = int(y1 - delta_height)
    x2 = int(x2 + delta_width)
    y2 = int(y2 + delta_height)
    # 扩大人脸区域的尺寸
    expand_ratio = 0.3  # 调整扩大比例
    dx = int((x2 - x1) * expand_ratio / 2)
    dy = int((y2 - y1) * expand_ratio / 2)
    x1 -= dx
    y1 -= dy
    x2 += dx
    y2 += dy
    # 在输出图片上把人脸部分用纯黑色覆盖
    image[y1:y2, x1:x2] = [0, 0, 0]
    # 输出调整后的坐标矩阵
    adjusted_coordinates = np.array([x1, y1, x2, y2])
    return adjusted_coordinates

# 遍历检测到的人脸
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        # 计算人脸边界框的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        box = adjusted_box(box)
        start_x, start_y, end_x, end_y = box.astype(np.int)

        # 提取人脸区域
        face = image[start_y:end_y, start_x:end_x]

        # 缩放人脸为128x128
        resized_face = cv2.resize(face, (128, 128))

        # 在输出图片中央放置缩小后的人脸
        face_h, face_w = resized_face.shape[:2]
        pos_x = (512 - face_w) // 2
        pos_y = (512 - face_h) // 2
        output_image[pos_y:pos_y+face_h, pos_x:pos_x+face_w] = resized_face
    break
cv2.imwrite("beauty_detected.jpg", output_image)
