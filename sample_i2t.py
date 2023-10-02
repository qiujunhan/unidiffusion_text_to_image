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
class image_to_caption_decoder():
    def __init__(self,config,nnet,clip_img_model,clip_img_model_preprocess,autoencoder):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_seed(config.seed)
        self.config = ml_collections.FrozenConfigDict(config)
        self._betas = self.stable_diffusion_beta_schedule()
        self.N = len(self._betas)

        self.nnet = nnet
        self.use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
        if self.use_caption_decoder:
            from libs.caption_decoder import CaptionDecoder
            self.caption_decoder = CaptionDecoder(device=self.device, **config.caption_decoder)
        else:
            self.caption_decoder = None
        self.clip_text_model = libs.clip.FrozenCLIPEmbedder(device=self.device, version=config.clip_text_model)
        self.clip_text_model.eval()
        self.clip_text_model.to(self.device)
        self.autoencoder = autoencoder
        self.autoencoder.to(self.device)
        self.clip_img_model, self.clip_img_model_preprocess = clip_img_model,clip_img_model_preprocess
        self.empty_context = self.clip_text_model.encode([''])[0]


    def stable_diffusion_beta_schedule(self,linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
        _betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
        return _betas.numpy()
    def set_seed(self,seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_contexts(self,img_path,config, clip_img_model, clip_img_model_preprocess, autoencoder):

        resolution = config.z_shape[-1] * 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        contexts = torch.randn(1, 77, config.clip_text_dim).to(device)





        from PIL import Image
        img_contexts = []
        clip_imgs = []

        def get_img_feature(image):
            image = np.array(image).astype(np.uint8)
            image = utils.center_crop(resolution, resolution, image)
            clip_img_feature = clip_img_model.encode_image(
                clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> 1 c h w')
            image = torch.tensor(image, device=device)
            moments = autoencoder.encode_moments(image)

            return clip_img_feature, moments

        image = Image.open(img_path).convert('RGB')
        clip_img, img_context = get_img_feature(image)

        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * 1
        clip_imgs = clip_imgs * 1

        img_contexts = torch.concat(img_contexts, dim=0)
        clip_imgs = torch.stack(clip_imgs, dim=0)

        return contexts, img_contexts, clip_imgs

    def i2t_nnet(self,x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)

        z_out, clip_img_out, text_out = self.nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=self.device, dtype=torch.int) + self.config.data_type)

        if self.config.sample.scale == 0.:
            return text_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = self.nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * self.N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)

        return text_out + self.config.sample.scale * (text_out - text_out_uncond)


    def img_decoder(self,img_path):
        self.nnet.eval()
        if self.config.get('benchmark', False):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        contexts, img_contexts, clip_imgs = self.prepare_contexts(img_path,self.config, self.clip_img_model,
                                                             self.clip_img_model_preprocess, self.autoencoder)

        z_img = self.autoencoder.sample(img_contexts)
        _n_samples = img_contexts.size(0)
        all_text = []

        def sample_fn( **kwargs):

            _z_init = torch.randn(_n_samples, *self.config.z_shape, device=self.device)
            _clip_img_init = torch.randn(_n_samples, 1, self.config.clip_img_dim, device=self.device)
            _text_init = torch.randn(_n_samples, 77, self.config.text_dim, device=self.device)

            _x_init = _text_init
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self._betas, device=self.device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * self.N
                return self.i2t_nnet(x, t, **kwargs)


            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            with torch.no_grad():
                with torch.autocast(device_type=self.device):
                    start_time = time.time()
                    x = dpm_solver.sample(_x_init, steps=self.config.sample.sample_steps, eps=1. / self.N, T=1.)
                    end_time = time.time()
                    print(
                        f'\ngenerate {_n_samples} samples with {self.config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')



            return x
        _text = sample_fn( z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
        all_text.append(_text)





        torch.save(torch.concatenate(all_text).mean(0).unsqueeze(0),img_path.replace(".png",".pt").replace(".jpg",".pt").replace(".jpeg",".pt"))









