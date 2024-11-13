#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recusion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"


# # image tasks
# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava_onevision \
#     --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
#     --tasks pope,mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \
#     --output_path ./logs/ \
#     --generation_type default \
#     # --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=woohyeon_ov_default"

# --tasks ai2d,chartqa,docvqa_val,infovqa_val,mme,realworldqa,mathvista_testmini,llava_in_the_wild,mmvet,mmbench_en_dev,ocrbench,mmmu,mathverse_testmini_vision_intensive,mathverse_testmini_vision_only,seedbench,scienceqa_img,mmstar \


python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type downsampled \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold 0.5 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type reduced \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-336-tot"