# coding=utf-8
import sys
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageOps
sys.path.append("BLIP")
from BLIP.models import blip

BLIP_PATH = os.path.abspath('BLIP')
img_size = 512
blip_model = blip.blip_decoder("huggingface/BLIP/model_base_caption_capfilt_large.pth",BertTokenizer_path = 'huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076',image_size=img_size, vit='base',med_config = os.path.join(BLIP_PATH,'configs','med_config.json'))

blip_model.eval()
blip_model = blip_model.to("cuda")

file = "processed_train_data/boy1/00000-0-00000-0-0.png"
img = Image.open(file)
img = ImageOps.exif_transpose(img)
img = img.convert("RGB")

gpu_image = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(img).unsqueeze(0).to("cuda")
caption = blip_model.generate(gpu_image, sample=True, num_beams=1, min_length=24, max_length=48)
print(caption[0])
