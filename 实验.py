import random

assist_prompts = [i.strip() for i in open("processed_train_data/assist_prompts.txt", "r").readlines()]
class_word = "man"
while True:
    input_prompt = random.choice(assist_prompts)
    input_prompt = input_prompt.format(class_word)