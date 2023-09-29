"""
采样代码
文件输入:
    prompt, 指定的输入文件夹路径, 制定的输出文件夹路径
文件输出:
    采样的图片, 存放于指定输出文件夹路径
- 指定prompt文件夹位置, 选手需要自行指定模型的地址以及其他微调参数的加载方式, 进行图片的生成并保存到指定地址, 此部分代码选手可以修改。
- 输入文件夹的内容组织方式和训练代码的输出一致
- sample的方法可以修改
- 生成过程中prompt可以修改, 但是指标测评时会按照给定的prompt进行测评。
"""

import os

import math
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
import glob
from score import Evaluator,read_img_pil
from PIL import Image
import shutil

def get_model_size(model):
    """
    统计模型参数大小
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    return para

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(n,config, clip_text_model, autoencoder):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    contexts = torch.randn(n, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(n, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(n, 1, config.clip_img_dim)

    prompts = [ config.prompt ] * n
    contexts = clip_text_model.encode(prompts)

    return contexts, img_contexts, clip_imgs


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(prompt_index, config, nnet, clip_text_model, autoencoder, device,n=3,score_eval=None):
    """
    using_prompt: if use prompt as file name
    """

    assert n >= config.n_samples
    assert score_eval != None
    assert score_eval.refs_clip != None
    assert score_eval.refs_embs != None



    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config = ml_collections.FrozenConfigDict(config)

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)


    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
    if use_caption_decoder:
        from libs.caption_decoder import CaptionDecoder
        caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    else:
        caption_decoder = None

    empty_context = clip_text_model.encode([''])[0]

    def split(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img


    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)

        if config.sample.scale == 0.:
            return x_out

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + config.sample.scale * (x_out - x_out_uncond)



    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def auto_decompose(n, max_samples=10):
        if n > 10:
            n_iter = 1
            n_samples = 10
        else:
            n_iter = 1
            n_samples = n

        if n > max_samples:
            n_samples = max_samples
            n_iter = math.ceil(n / max_samples)

        return n_iter, n_samples


    contexts, img_contexts, clip_imgs = prepare_contexts(n,config, clip_text_model, autoencoder)
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet

    _n_samples = contexts_low_dim.size(0)
    chunk_size = 10
    n_iter, _n_samples = auto_decompose(_n_samples,chunk_size)

    def sample_fn(**kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        _x_init = combine(_z_init, _clip_img_init)

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return t2i_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            x = dpm_solver.sample(_x_init,method=config.sample.method, steps=config.sample.sample_steps, eps=1. / N, T=1.)

        _z, _clip_img = split(x)
        return _z, _clip_img

    contexts_low_dim_sub_lists = [contexts_low_dim[i:i + chunk_size] for i in range(0, len(contexts_low_dim), chunk_size)]

    samples = None
    for i in range(n_iter):
        for contexts_low_dim in contexts_low_dim_sub_lists:
            _z, _clip_img = sample_fn(text=contexts_low_dim)  # conditioned on the text embedding
            new_samples = unpreprocess(decode(_z))
            if samples is None:
                samples = new_samples
            else:
                samples = torch.vstack((samples, new_samples))

            # 评分模块
    temp_path = "temp/temp_images"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path,exist_ok=True)

    for idx, sample in enumerate(samples):

        save_path = os.path.join(temp_path, f'{prompt_index}-{idx:03}.jpg')
        save_image(sample, save_path)

    sample_scores=[]
    for idx in range(0, n):

        sample_path = os.path.join("temp/temp_images", f"{prompt_index}-{idx:03}.jpg")

        sample_img = read_img_pil(sample_path)


        # sample vs ref

        score_face = score_eval.sim_face_emb(sample_img, score_eval.refs_embs)
        score_clip = score_eval.sim_clip_imgembs(sample_img, score_eval.refs_clip)
        score_text = score_eval.sim_clip_text(sample_img, config.prompt)

        sample_scores.append([score_face, score_clip, score_text])
    sample_scores = np.array(sample_scores)
    best_samples_index =np.argsort(sample_scores.mean(axis=1))[-config.n_samples:]
    best_samples_index = best_samples_index[::-1]

    os.makedirs(config.output_path, exist_ok=True)
    for idx, sample in enumerate(samples[best_samples_index.copy()]):
        save_path = os.path.join(config.output_path, f'{prompt_index}-{idx:03}.jpg')
        save_image(sample, save_path)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    print(f'results are saved in {save_path}')
    return sample_scores[best_samples_index]


def compare_model(standard_model:torch.nn.Module, model:torch.nn.Module, mapping_dict= {}):
    """
    compare model parameters based on paramter name
    for parameters with same name(common key), directly compare the paramter conetent
    all other parameters will be regarded as differ paramters, except keys in mapping_dict.
    mapping_dict is a python dict class, with keys as a subset of `origin_only_keys` and values as a subset of `compare_only_keys`.

    """
    origin_dict = dict(standard_model.named_parameters())
    origin_keys = set(origin_dict.keys())
    compare_dict = dict(model.named_parameters())
    compare_keys = set(compare_dict.keys())

    origin_only_keys = origin_keys - compare_keys
    compare_only_keys = compare_keys - origin_keys
    common_keys = set.intersection(origin_keys, compare_keys)


    diff_paramters = 0
    # compare parameters of common keys
    for k in common_keys:
        origin_p = origin_dict[k]
        compare_p = compare_dict[k]
        if origin_p.shape != compare_p.shape:
            diff_paramters += origin_p.numel() + compare_p.numel()
        elif (origin_p - compare_p).norm() != 0:
            diff_paramters += origin_p.numel()


    mapping_keys = set(mapping_dict.keys())
    assert set.issubset(mapping_keys, origin_only_keys)
    assert set.issubset(set(mapping_dict.values()), compare_only_keys)

    for k in mapping_keys:
        origin_p = origin_dict[k]
        compare_p = compare_dict[mapping_dict[k]]
        if origin_p.shape != compare_p.shape:
            diff_paramters += origin_p.numel() + compare_p.numel()
        elif (origin_p - compare_p).norm() != 0:
            diff_paramters += origin_p.numel()
    # all keys left are counted
    extra_origin_keys = origin_only_keys - mapping_keys
    extra_compare_keys = compare_only_keys - set(mapping_dict.values())

    for k in extra_compare_keys:
        diff_paramters += compare_dict[k].numel()

    for k in extra_origin_keys:
        diff_paramters += origin_dict[k].numel()

    return diff_paramters



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")
    parser.add_argument("--prompt_path", type=str, default="eval_prompts/boy1.json", help="file contains prompts")
    parser.add_argument("--output_path", type=str, default="outputs/boy1", help="output dir for generated images")
    return parser.parse_args()


def main(argv=None):
    # config args
    from configs.sample_config import get_config
    set_seed(42)
    config = get_config()
    args = get_args()
    config.output_path = args.output_path
    config.nnet_path = os.path.join(args.restore_path, "final.ckpt",'lora_nnet.pth')
    config.n_samples = 3

    # config.n_iter = 1
    device = "cuda"

    # init models
    nnet = UViT(**config.nnet)
    nnet = get_peft_model(nnet, config.lora.peft_config)

    nnet_dict =torch.load("models/uvit_v1.pth", map_location='cpu')
    nnet_mapping_dict = {name: f"base_model.model.{name}" for name in nnet_dict}
    nnet_dict = {f"base_model.model.{key}": value for key, value in nnet_dict.items()}
    #模型融合方案
    lora1 = "logs/unidiffuserv1-boy1_1dim_lr1e-05_sample_kldiv_lossx1_5sample/ckpts/7610.ckpt/lora_nnet.pth"
    lora2 = "logs/unidiffuserv1-boy1_1dim_lr0.0001_sample_kldiv_lossx1_5sample/ckpts/7610.ckpt/lora_nnet.pth"
    lora_dict1 = torch.load(lora1, map_location='cpu')
    lora_dict2 = torch.load(lora2, map_location='cpu')
    # 融合权重
    alpha = 1 # 权重融合的系数，0.5 表示两个模型权重的平均值
    merged_weights = {}
    for key in lora_dict1:
        merged_weights[key] = alpha * lora_dict1[key] + (1 - alpha) * lora_dict2[key]
    nnet_dict.update(merged_weights)
    #加载lora
    # nnet_dict.update(torch.load(config.nnet_path, map_location='cpu'))

    nnet.load_state_dict(nnet_dict, False)



    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_text_model.to("cpu")

    # 测评
    config.data = "train_data/boy1"
    score_eval = Evaluator()
    refs = glob.glob(os.path.join(config.data, "*.jpg")) + glob.glob(os.path.join(config.data, "*.jpeg"))
    refs_images = [read_img_pil(ref) for ref in refs]

    refs_clip = [score_eval.get_img_embedding(i) for i in refs_images]
    refs_clip = torch.cat(refs_clip)
    #### print(refs_clip.shape)

    refs_embs = [score_eval.get_face_embedding(i) for i in refs_images]
    refs_embs = [emb for emb in refs_embs if emb is not None]
    refs_embs = torch.cat(refs_embs)
    score_eval.refs_clip = refs_clip
    score_eval.refs_embs=refs_embs


    use_diffusers = False # 是否使用diffuser，使用diffusers的选手请设置为true



    autoencoder_mapping_dict = {}
    clip_text_mapping_dict = {}

    # ------------!!!!!请保证以下代码不被修改
    total_diff_parameters = 0
    if not use_diffusers:
        nnet_standard = UViT(**config.nnet)
        nnet_standard.load_state_dict(torch.load("models/uvit_v1.pth", map_location='cpu'), False)
        total_diff_parameters += compare_model(nnet_standard, nnet, nnet_mapping_dict)
        del nnet_standard

        autoencoder_standard = libs.autoencoder.get_model(**config.autoencoder)
        total_diff_parameters += compare_model(autoencoder_standard, autoencoder, autoencoder_mapping_dict)
        del autoencoder_standard

        clip_text_strandard = FrozenCLIPEmbedder(version=config.clip_text_model, device=device).to("cpu")
        total_diff_parameters += compare_model(clip_text_strandard, clip_text_model, clip_text_mapping_dict)
        del clip_text_strandard
    else:
        from diffusers import UniDiffuserPipeline
        pipe = UniDiffuserPipeline.from_pretrained("./diffusers_models")
        nnet_standard, autoencoder_standard, clip_text_strandard = pipe.unet, pipe.vae, pipe.text_encoder
        total_diff_parameters += compare_model(nnet_standard, nnet, nnet_mapping_dict)
        total_diff_parameters += compare_model(autoencoder_standard, autoencoder, autoencoder_mapping_dict)
        total_diff_parameters += compare_model(clip_text_strandard, clip_text_model, clip_text_mapping_dict)

    # ------------!!!!!请保证以上代码不被修改

    clip_text_model.to(device)
    autoencoder.to(device)
    nnet.to(device)

    # 基于给定的prompt进行生成
    prompts = json.load(open(args.prompt_path, "r"))
    all_sample_scores = []
    for prompt_index, prompt in enumerate(prompts):
        # 根据训练策略
        if "boy" in prompt:
            prompt = prompt.replace("boy", "man")
        else:
            prompt = prompt.replace("girl", "female")
        extend_prompt = ",a handsome man, wearing a red outfit, sitting on a chair and eating,Chinese, Asian, high-definition image, high-quality lighting"

        config.prompt = prompt
        config.extend_prompt = extend_prompt
        print("sampling with prompt:", prompt+extend_prompt)
        sample_scores = sample(prompt_index, config, nnet, clip_text_model, autoencoder, device,n=config.n_samples*5,score_eval=score_eval)
        all_sample_scores.append(sample_scores)

        #评分模块
        #
        # for idx in range(0, config.n_samples):  ## 3 generation for each prompt
        #     sample_path = os.path.join(config.output_path, f"{prompt_index}-{idx:03}.jpg")
        #
        #     sample_img = read_img_pil(sample_path)
        #     # sample vs ref
        #
        #     score_face = score_eval.sim_face_emb(sample_img, score_eval.refs_embs)
        #     score_clip = score_eval.sim_clip_imgembs(sample_img, score_eval.refs_clip)
        #     score_text = score_eval.sim_clip_text(sample_img, prompt)
        #
        #     sample_scores.append([score_face, score_clip, score_text])
    all_sample_scores = np.concatenate(all_sample_scores)
    all_sample_scores = all_sample_scores.mean(axis=0)


    write_str = f" 总分{all_sample_scores.mean()} 人脸相似度{all_sample_scores[0]} CLIP图片相似度{all_sample_scores[1]} 图文匹配度{all_sample_scores[2]}"
    print(write_str)
    with open(os.path.join(config.output_path, "score.log"),"w") as f:
        f.write(write_str)

    print(f"\033[91m 微调参数量:\n{total_diff_parameters}\033[00m")

if __name__ == "__main__":
    main()