import ml_collections
from peft import LoraConfig, TaskType,get_peft_model

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1
    config.gradient_accumulation_steps = 1
    config.log_interval = 10
    config.eval_interval = 200
    config.save_interval = 300
    config.max_step = 10000
        
    config.num_workers = 0
    config.batch_size = 1
    config.resolution = 512
    
    config.clip_img_model = "ViT-B/32"
    # config.clip_text_model = "openai/clip-vit-large-patch14"
    config.clip_text_model = "huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
    config.only_load_model = True
    

    config.optimizer = d(
        name='adamw',
        lr=1e-4,
        weight_decay=0.03,
        betas=(0.9, 0.9),
        amsgrad=False
    )

    config.lr_scheduler = d(
        # name='customized',
        name='cosine_with_restarts',
        num_warmup_steps=0, num_training_steps=config.get_ref('max_step'), num_cycles=1
    )

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim'),
        tokenizer_path = "./models/gpt2"
    )

    config.nnet = d(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )
    config.lora =d(

    # target_modules = ["text_embed", "text_out", "clip_img_embed", "clip_img_out", "qkv",
    #                   "proj", "fc1", "fc2"],
    target_modules = ["qkv","proj"],

    peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                        lora_dropout=0,
                                        target_modules=[])
    )
    config.lora.peft_config.target_modules = config.lora.target_modules


    # sample
    config.mode = "t2i"
    config.n_samples = 2
    config.n_iter = 6
    config.nrow = 4
    config.sample = d(
        sample_steps=30,
        scale=7.,
        t2i_cfg_mode='true_uncond'
    )

    return config
