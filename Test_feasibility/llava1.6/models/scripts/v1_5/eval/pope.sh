#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-84-zero.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --positional_embedding_type zero \
#     --resized_image_size 84

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-84-zero.jsonl

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.6-7b-recursion-05.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --recursion_type all \
    --recursion_threshold 0.5

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.6-7b-recursion-05.jsonl
