"""
推理速度和内存使用标准程序
给定模型,随机种子,采样步数以及采样方法以获取固定数量的图片
在一定的差异容忍度下测量生成图片需要的时间以及显存占用情况
注意: 程序速度和显存优化并非赛题的主要部分, 分数权重待定, 请赛手做好各个子任务之间的平衡
"""


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
from PIL import Image
import clip
import time
from libs.clip import FrozenCLIPEmbedder
import numpy as np
from libs.uvit_multi_post_ln_v1 import UViT
from libs.caption_decoder import CaptionDecoder




def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)

    prompts = [ config.prompt ] * config.n_samples
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


def sample(config, nnet, clip_text_model, autoencoder, clip_img_model, 
           clip_img_model_preprocess, caption_decoder, device):
    """
    using_prompt: if use prompt as file name
    """
    n_iter = config.n_iter
    use_caption_decoder = True
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config = ml_collections.FrozenConfigDict(config)


    ############ start timing #############
    start_time = time.time()
    
    
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)


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
    def decode(_batch):
        return autoencoder.decode(_batch)


    contexts, img_contexts, clip_imgs = prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder)
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    _n_samples = contexts_low_dim.size(0)


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
            x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)

        _z, _clip_img = split(x)
        return _z, _clip_img


    samples = None    
    for i in range(n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim)  # conditioned on the text embedding
        new_samples = unpreprocess(decode(_z))
        if samples is None:
            samples = new_samples
        else:
            samples = torch.vstack((samples, new_samples))

    ############# end timing ##############
    end_time = time.time()

    # 文件需要保存为jpg格式
    os.makedirs(config.output_path, exist_ok=True)
    for idx, sample in enumerate(samples):
        save_path = os.path.join(config.output_path, f'{config.prompt}-{idx:03}.jpg')
        save_image(sample, save_path)
        

    print(f'\nresults are saved in {os.path.join(config.output_path)} :)')
    mem_use = torch.cuda.max_memory_reserved()
    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
    print(f"\nusing time: {end_time - start_time:.2f}s")
    return (mem_use, end_time - start_time)



def assert_same(path1, path2):
    img1_list = os.listdir(path1)
    img2_list = os.listdir(path2)

    if len(img1_list) != len(img2_list):
         return False
        
    def eval_single(img1, img2):
        img1 = np.array(Image.open(img1))
        img2 = np.array(Image.open(img2))
        mean_diff = np.linalg.norm(img1 - img2)/(512*512)
        if mean_diff < 1:
            return True
        else:
            return False

    for img1, img2 in zip(img1_list, img2_list):
        if eval_single(os.path.join(path1, img1), os.path.join(path2, img2)):
            continue
        else:
            return False
    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnet_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")
    parser.add_argument("--output_path", type=str, default="bench_samples", help="path to place output imgs")
    parser.add_argument("--half", action="store_true", help="half precision for memory optiomization")
    return parser.parse_args()


def main(argv=None):
    # config args
    from configs.sample_config import get_config
    set_seed(42)
    config = get_config()
    args = get_args()
    config.nnet_path = args.nnet_path
    config.output_path = args.output_path

    config.n_samples = 2
    config.n_iter = 15
    device = "cuda"

    # init models
    nnet = UViT(**config.nnet)
    print(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), False)
    nnet.to(device)
    if args.half:
        nnet = nnet.half()
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    config.prompt = "an elephant under the sea"

    sample(config, nnet, clip_text_model, autoencoder, clip_img_model, 
           clip_img_model_preprocess, caption_decoder, device)
    
    if assert_same("bench_samples_standard", config.output_path):
        print("error assertion passed")
    else:
        print("error assertion failed")
    

if __name__ == "__main__":
    main()