# coding=utf-8
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def blip(img_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # processor = BlipProcessor.from_pretrained("./pretrained_models/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("./pretrained_models/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    raw_image = Image.open(img_path).convert('RGB')

    # conditional image captioning
    text = "several people"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def main():
    img_path = "train_data/boy1/0.jpeg"
    blip(img_path)

if __name__ == "__main__":
    main()
