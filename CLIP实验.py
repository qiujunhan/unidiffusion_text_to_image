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
from glob import glob
import random

from configs.unidiffuserv1 import get_config
config = get_config()
config_name = "unidiffuserv1"



data_name = Path(config.data).stem

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
config.workdir = os.path.join(config.log_dir, f"{config_name}-{data_name}_{config.suffix}")
config.ckpt_root = os.path.join(config.workdir, 'ckpts')
config.meta_dir = os.path.join(config.workdir, "meta")


score_eval = TrainEvaluator()

score_eval.clip_model, score_eval.clip_preprocess = clip.load(config.clip_img_model, jit=False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
refs = glob(os.path.join(config.data, "*.jpg")) + glob(os.path.join(config.data, "*.jpeg"))
refs_images = [read_img_pil(ref) for ref in refs]

refs_clip = [score_eval.get_img_embedding(i) for i in refs_images]
refs_clip = torch.cat(refs_clip)
#### print(refs_clip.shape)

refs_embs = [score_eval.get_face_embedding(i) for i in refs_images]
refs_embs = [emb for emb in refs_embs if emb is not None]
refs_embs = torch.cat(refs_embs)
score_eval.refs_clip = refs_clip
score_eval.refs_embs = refs_embs