#pip install git+https://github.com/huggingface/transformers.git
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/6e723d92ee91ebcee4ba74d7017632f11ff4217b")
model = Blip2ForConditionalGeneration.from_pretrained("huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/6e723d92ee91ebcee4ba74d7017632f11ff4217b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
from PIL import Image

url = "processed_train_data/boy1/00000-0-00000-0-0.png"
image = Image.open(url).convert('RGB')


inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)