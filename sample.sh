#!/bin/bash
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_sim.json --output_path outputs/boy1_sim
#python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_sim.json --output_path outputs/boy2_sim
#python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_sim.json --output_path outputs/girl1_sim
#python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_sim.json --output_path outputs/girl2_sim
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_edit.json --output_path outputs/boy1_edit
#python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_edit.json --output_path outputs/boy2_edit
#python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_edit.json --output_path outputs/girl1_edit
#python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_edit.json --output_path outputs/girl2_edit
