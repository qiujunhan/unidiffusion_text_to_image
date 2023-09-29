
import os
import ml_collections
import torch
import random
import argparse
import utils
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import numpy as np
import clip
import time
from libs.clip import FrozenCLIPEmbedder
import numpy as np
import json
from libs.uvit_multi_post_ln_v1 import UViT
from peft import LoraConfig, TaskType,get_peft_model
from configs.sample_config import get_config
config = get_config()


lora1 = "logs/unidiffuserv1-boy1_1dim_lr0.0001_sample_kldiv_lossx1_5sample/ckpts/2810.ckpt"
lora2 = "logs/unidiffuserv1-boy1_1dim_lr0.0001_sample_kldiv_lossx1_5sample/ckpts/7610.ckpt"

lora_dict1 = torch.load(lora1, map_location='cpu')
lora_dict2 = torch.load(lora2, map_location='cpu')

# 融合权重
alpha = 0.5  # 权重融合的系数，0.5 表示两个模型权重的平均值
merged_weights = {}
for key in lora_dict1:
    merged_weights[key] = alpha * lora_dict1[key] + (1 - alpha) * lora_dict2[key]


# init models
nnet = UViT(**config.nnet)
nnet = get_peft_model(nnet, config.lora.peft_config)
nnet_dict =torch.load("models/uvit_v1.pth", map_location='cpu')
nnet_dict = {f"base_model.model.{key}": value for key, value in nnet_dict.items()}


