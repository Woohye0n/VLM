#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recusion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"


## run in default setting
# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type default \
#     --fix_grid default \
#     --verbosity INFO 
    #--wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_default" 

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
    --generation_type recursion_retain_base \
    --fix_grid 2x2 \
    --attention_thresholding_type confidence_based_topk\
    --attention_threshold 0.3 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=pope_recursive_retain_base_Det-True_sensitivte_confidence_based_top30"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type confidence_based_topk\
#     --attention_threshold 0.5 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=pope_recursive_Det-None_sensitivte_confidence_based_top50"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold 0.3 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity INFO \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_recursive_0.3_eos"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold 0.5 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity INFO \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_recursive_0.5_eos"

    
    
    
    