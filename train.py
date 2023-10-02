"""
训练代码
代码输入:
    - 数据文件夹路径, 其中包含近近脸照文件夹和全身照文件夹, 
    - 指定的输出路径, 用于输出模型
    - 其他的参数需要选手自行设定
代码输出:
    - 微调后的模型以及其他附加的子模块
"""

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
from process_data import DataProcessing
import time
import re

def train(config):
    
    """
    prepare models
    准备各类需要的模型
    """
    accelerator, device = utils.setup(config)
    
    train_state = utils.initialize_train_state(config, device, uvit_class=UViT)
    logging.info(f'load nnet from {config.nnet_path}')
    if config.resume != None:
        train_state.resume(config.resume)
    #加载权重
    if config.mode == "sim":
        # train_state.resume( ckpt_path=os.path.join("logs/unidiffuserv1-boy1_1dim_lr0.0001_usei2t_only_proj_裁剪加原图/ckpts/",config.mode,"0040.ckpt"))
        pass
    elif config.mode == "edit":
        # train_state.resume( ckpt_path=os.path.join("logs/unidiffuserv1-boy1_1dim_lr0.0001_usei2t_only_proj_裁剪加原图/ckpts/",config.mode,"0020.ckpt"))
        pass
    # caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)

    nnet, optimizer = accelerator.prepare(train_state.nnet, train_state.optimizer)
    nnet.to(device)
    lr_scheduler = train_state.lr_scheduler

    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)



    score_eval = TrainEvaluator()
    score_eval.clip_model, score_eval.clip_preprocess = clip_img_model,clip_img_model_preprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    #图片预处理
    i2t = image_to_caption_decoder(config,nnet=nnet,clip_img_model=clip_img_model,clip_img_model_preprocess=clip_img_model_preprocess,autoencoder=autoencoder)
    data_preprocess = DataProcessing(data_path=config.data,i2t=i2t,mode=config.mode)
    data_preprocess.process()


    refs = glob(os.path.join(config.data, "*.jpg")) + glob(os.path.join(config.data, "*.jpeg"))
    refs_images = [read_img_pil(ref) for ref in refs]

    refs_clip = [score_eval.get_img_embedding(i) for i in refs_images]
    refs_clip = torch.cat(refs_clip)
    #### print(refs_clip.shape)

    refs_embs = [score_eval.get_face_embedding(i) for i in refs_images]
    refs_embs = [emb for emb in refs_embs if emb is not None]
    refs_embs = torch.cat(refs_embs)
    assist_prompts = [i.strip() for i in open("processed_train_data/assist_prompts.txt", "r").readlines()]
    score_eval.refs_clip = refs_clip
    score_eval.refs_embs = refs_embs
    """
    处理数据部分
    """
    # process data


    train_dataset = PersonalizedBase(config.data,config=config, resolution=512, class_word="man" if "boy" in config.data else "girl")
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      drop_last=True
                                      )

    train_data_generator = utils.get_data_generator(train_dataset_loader, enable_tqdm=accelerator.is_main_process, desc='train')

    logging.info("saving meta data")
    os.makedirs(config.meta_dir, exist_ok=True)
    with open(os.path.join(config.meta_dir, "config.yaml"), "w") as f:
        f.write(yaml.dump(config))
        f.close()
    
    _betas = stable_diffusion_beta_schedule()
    schedule = Schedule(_betas)
    logging.info(f'use {schedule}')

    def train_step(cache_text,cache_img,cache_img4clip,sample_loss):
        metrics = dict()
        img, img4clip, text, data_type = next(train_data_generator)
        img = img.to(device)
        img4clip = img4clip.to(device)
        data_type = data_type.to(device)
        text = text.to(device)
        with torch.no_grad():
            hash_img = hash(img)
            if hash_img not in cache_img:
                cache_img[hash_img] = autoencoder.encode(img)
                cache_img4clip[hash_img] = clip_img_model.encode_image(img4clip).unsqueeze(1)
                # text = clip_text_model.encode(text)
                # cache_text[hash_img] = caption_decoder.encode_prefix(text)
                z = cache_img[hash_img]
                clip_img = cache_img4clip[hash_img]
                # text = cache_text[hash_img]
            else:
                z = cache_img[hash_img]
                clip_img = cache_img4clip[hash_img]
                # text = cache_text[hash_img]
            # z = autoencoder.encode(img)
            # clip_img = clip_img_model.encode_image(img4clip).unsqueeze(1)
            # text = clip_text_model.encode(text)
            # text = caption_decoder.encode_prefix(text)

        loss, loss_img, loss_clip_img, loss_text = LSimple_T2I(img=z, clip_img=clip_img, text=text, data_type=data_type, nnet=nnet, schedule=schedule,sample_loss=sample_loss, device=device)

        accelerator.backward(loss.mean())
        optimizer.step()
        print(train_state.optimizer.param_groups[0]['lr'])
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        optimizer.zero_grad()
        metrics['loss'] = accelerator.gather(loss.detach().mean()).mean().item()
        metrics['loss_img'] = accelerator.gather(loss_img.detach().mean()).mean().item()
        metrics['loss_clip_img'] = accelerator.gather(loss_clip_img.detach().mean()).mean().item()
        metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = train_state.optimizer.param_groups[0]['lr']
        return metrics

    @torch.no_grad()
    @torch.autocast(device_type='cuda')
    def eval(total_step,metrics):
        """
        write evaluation code here
        """
        from configs.sample_config import get_config as get_sample_config
        sample_config = get_sample_config()
        sample_config.n_samples=config.eval_samples
        sample_config.n_iter = 1
        sample_config.sample.sample_steps=20
        input_prompt = "a {}, wearing a red outfit, sitting on a chair and eating"
        class_word = "man" if "boy" in config.data else "girl"
        # input_prompt = random.choice(assist_prompts)
        input_prompt = input_prompt.format(class_word)
        add_prompt=",Chinese, Asian, lifestyle photo, photography shot, high-definition image, high-quality lighting, visible facial features, single image, non-collage."
        change_prompt = input_prompt+add_prompt
        sample_config.prompt = change_prompt

        config.sample_root = os.path.join(config.workdir, 'sample_root')
        os.makedirs(config.sample_root, exist_ok=True)
        sample_config.output_path = config.sample_root
        print("sampling with prompt:", change_prompt)
        from sample import sample
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sample(total_step, sample_config, nnet, clip_text_model, autoencoder, device,n=sample_config.n_samples*config.n_select,score_eval=score_eval,chunk_size = 10)
        sample_scores = []
        sample_loss = []
        for idx in range(0, sample_config.n_samples):  ## 3 generation for each prompt
            sample_path = os.path.join(config.sample_root, f"{total_step}-{idx:03}.jpg")

            sample = read_img_pil(sample_path)
            # sample vs ref

            face_loss,score_face = score_eval.sim_face_emb(sample, refs_embs,train=True)
            clip_loss,score_clip = score_eval.sim_clip_imgembs(sample, refs_clip,train=True)
            text_loss,score_text = score_eval.sim_clip_text(sample, input_prompt,train=True)
            sample_loss.append([face_loss*0.5,clip_loss*0,text_loss*2])
            sample_scores.append([score_face, score_clip, score_text])
            os.rename(sample_path, os.path.join(config.sample_root, f"{config.mode}_{total_step}-{idx:03}.jpg"))
        sample_scores = np.array(sample_scores)
        sample_loss = torch.tensor(sample_loss)
        sample_scores = sample_scores.mean(axis =0)

        write_str = f"total_step: {total_step} 总分{sample_scores.mean()} 人脸相似度{sample_scores[0]} CLIP图片相似度{sample_scores[1]} 图文匹配度{sample_scores[2]} "
        print(write_str)




        return sample_scores,sample_loss

    def loop():

        log_step = train_state.step + config.eval_interval
        eval_step = train_state.step + config.eval_interval
        save_step = train_state.step + config.save_interval
        cache_text = {}
        cache_img = {}
        cache_img4clip = {}

        sample_loss = torch.tensor([0.,0.,0.], requires_grad=True)
        sample_scores = torch.tensor([0., 0., 0.], requires_grad=True)
        while True:

            nnet.train()
            with accelerator.accumulate(nnet):

                metrics = train_step(cache_text,cache_img,cache_img4clip,sample_loss)

            if accelerator.is_main_process:
                nnet.eval()
                total_step = train_state.step * config.batch_size
                if total_step >= eval_step or total_step  == config.max_step:
                    sample_scores,sample_loss = eval(total_step,metrics)
                    sample_scores = torch.tensor(sample_scores, requires_grad=True)

                    # last_save_step = max(config.save_interval,min(total_step,save_step))

                    train_state.add_sample(save_step,sample_scores)
                    eval_step += config.eval_interval
                if total_step >= log_step or total_step  == config.max_step:
                    logging.info(utils.dct2str(dict(step=total_step, **metrics)))
                    wandb.log(utils.add_prefix(metrics, 'train'), step=total_step)
                    write_str = f"total_step: {total_step} 总分{sample_scores.mean()} 人脸相似度{sample_scores[0]} CLIP图片相似度{sample_scores[1]} 图文匹配度{sample_scores[2]} "

                    with open(os.path.join(config.sample_root, "score.log"), "a") as f:
                        f.write(write_str + "\n")

                    log_step += config.log_interval



                if total_step >= save_step :
                    logging.info(f'Save and eval checkpoint {total_step}...')
                    train_state.save(os.path.join(config.ckpt_root, f'{total_step:04}.ckpt'))
                    
                    save_step += config.save_interval

            accelerator.wait_for_everyone()
            
            if total_step  >= config.max_step:
                logging.info(f"saving final ckpts to {config.outdir}...")
                # <<<notice>>> 只需要保存被优化的部分的参数即可，不必保存整个模型
                train_state.save(os.path.join(config.ckpt_root, f'{total_step:04}.ckpt'))

                #开始分析log，择优选择模型保存到model_output

                best_source_path,now_step = train_state.copy_best_ckpt(log_path=config.ckpt_root,final_path=os.path.join(config.outdir, 'final.ckpt',config.mode),config=config)
                return best_source_path,now_step


    best_source_path,now_step = loop()
    return best_source_path,now_step




def get_args():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument('-d', '--data', type=str, default="train_data/boy1", help="datadir")
    parser.add_argument('-o', "--outdir", type=str, default="model_ouput/boy1", help="output of model")
    
    # args of logging
    parser.add_argument("--logdir", type=str, default="logs", help="the dir to put logs")
    parser.add_argument("--nnet_path", type=str, default="models/uvit_v1.pth", help="nnet path to resume")

    
    return parser.parse_args()

def main():
    # 赛手需要根据自己的需求修改config file
    from configs.unidiffuserv1 import get_config

    modes = ["sim", "edit",]
    for mode in modes:
        if mode == "sim":
            config = get_config(max_step=10000)
            config.mode="sim"
            config.lora.peft_config.target_modules = config.lora.target_modules_sim

        elif mode == "edit":
            config = get_config(max_step=4500)
            config.mode = "edit"
            config.lora.peft_config.target_modules = config.lora.target_modules_edit



        config_name = "unidiffuserv1"
        args = get_args()
        config.log_dir = args.logdir
        config.outdir = args.outdir
        config.data = args.data
        data_name = Path(config.data).stem
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        config.workdir = os.path.join(config.log_dir, f"{config_name}-{data_name}_{config.suffix}")

        config.ckpt_root = os.path.join(config.workdir, 'ckpts', config.mode)
        config.meta_dir = os.path.join(config.workdir, "meta")
        config.nnet_path = args.nnet_path
        os.makedirs(config.workdir, exist_ok=True)
        config.resume = None
        #粗调到精调
        while config.save_interval > 1:
            best_source_path,now_step = train(config)
            config.max_step = now_step + config.save_interval
            config.save_interval //=2
            config.eval_interval //=2
            config.resume = best_source_path





if __name__ == "__main__":
    main()
