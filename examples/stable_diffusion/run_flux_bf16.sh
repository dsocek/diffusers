#!/bin/bash
#model=black-forest-labs/FLUX.1-dev
model=../../../FLUX.1-dev
python text_to_image_generation.py \
     --model_name_or_path $model \
     --prompts "A cat holding a sign that says hello world" \
     --num_images_per_prompt 10 \
     --batch_size 1 \
     --num_inference_steps 30 \
     --image_save_dir ./flux_1_images \
     --scheduler flow_match_euler_discrete \
     --bf16 \
     --use_torch_compile
