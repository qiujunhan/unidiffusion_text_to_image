import torch
import utils
from absl import logging
import os
import wandb
import libs.autoencoder
import clip
from libs.clip import FrozenCLIPEmbedder
from libs.caption_decoder import CaptionDecoder
from torch.utils.data import DataLoader
from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
import argparse
import yaml
import datetime
from pathlib import Path
from libs.data import PersonalizedBase
from libs.uvit_multi_post_ln_v1 import UViT
from sample_i2t import image_to_caption_decoder
from score import TrainEvaluator,read_img_pil
import gc
import glob
import random
import numpy as np
import time

import ml_collections
import torch
import random
import utils
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
import clip
from PIL import Image
import time
from glob import glob
from libs.uvit_multi_post_ln_v1 import UViT
import os
import cv2


class DataProcessing():
    def __init__(self,data_path,i2t,mode):
        self.data_path = data_path
        self.i2t = i2t
        self.processed_path = r'./processed_train_data/'
        self.mode = mode
        prototxt_path = "./huggingface/deploy.prototxt.txt"
        model_path = "./huggingface/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        self.dir_list = os.listdir(self.data_path)
        self.name = os.path.basename(self.data_path)
        self.dir_list = [os.path.join(self.data_path, file_path) for file_path in self.dir_list if not file_path.endswith(".ipynb_checkpoints")]
    def process_image(self,):
        if self.mode == "edit":

            output_dir = os.path.join(self.processed_path, self.name, "edit")

        elif self.mode == "sim":
            output_dir = os.path.join(self.processed_path, self.name, "sim")
        else:
            raise
        if os.path.exists(output_dir):
            print("预处理文件已存在")
            return
        for dir in self.dir_list:
            pil_image = Image.open(dir).convert("RGB")
            os.makedirs(output_dir,exist_ok=True)
            save_crop_path = os.path.join(output_dir,os.path.basename(dir).split(".")[0]+"_crop.png")
            save_path = os.path.join(output_dir,os.path.basename(dir).split(".")[0]+".png")
            pil_image.save(save_crop_path)
            if self.mode == "edit":
                pil_image.save(save_path)
                self.i2t.img_decoder(save_path)
                self.face_extraction(save_crop_path)
                self.i2t.img_decoder(save_crop_path)
            elif self.mode == "sim":
                self.face_extraction(save_crop_path)
                self.i2t.img_decoder(save_crop_path)

    def face_extraction(self,image_path):

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
            dy = int((y2 - y1) * expand_ratio )
            x1 -= dx
            y1 -= dy*0.75
            x2 += dx
            y2 += dy*0.25
            # # 在输出图片上把人脸部分用纯黑色覆盖
            # image[y1:y2, x1:x2] = [0, 0, 0]
            # 输出调整后的坐标矩阵
            adjusted_coordinates = np.array([x1, y1, x2, y2])
            return adjusted_coordinates

        # 加载输入图片并获取尺寸
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        # 进行人脸检测
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # 创建一个白色的512x512图片
        output_image = np.ones((512, 512, 3), np.uint8) * 255
        # 遍历检测到的人脸
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # 计算人脸边界框的坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = adjusted_box(box)
                start_x, start_y, end_x, end_y = box.astype(int)

                # 提取人脸区域
                face = image[start_y:end_y, start_x:end_x]

                resized_face = cv2.resize(face, (512, 512))

                # 在输出图片中央放置缩小后的人脸
                face_h, face_w = resized_face.shape[:2]
                pos_x = (512 - face_w) // 2
                pos_y = (512 - face_h) // 2
                output_image[pos_y:pos_y + face_h, pos_x:pos_x + face_w] = resized_face
                cv2.imwrite(image_path, output_image)
            break


    def process(self,):
        self.process_image()


        


