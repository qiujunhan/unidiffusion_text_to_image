#pip install git+https://github.com/huggingface/transformers.git
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from glob import glob

from PIL import Image
processor = AutoProcessor.from_pretrained("huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/6e723d92ee91ebcee4ba74d7017632f11ff4217b")
model = Blip2ForConditionalGeneration.from_pretrained("huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/6e723d92ee91ebcee4ba74d7017632f11ff4217b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
for path in glob("processed_train_data/boy1/*.png"):

    image = Image.open(path).convert('RGB')

    prompts = ["Question: Which region is this person from?? Answer:",
               "Question: Please provide a detailed description of this image, accurately describing every detail? Answer:",
               ]
    print(path)
    for prompt in prompts:
        inputs = processor(image,text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(generated_text)