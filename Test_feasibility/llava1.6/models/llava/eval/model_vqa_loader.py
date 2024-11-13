import argparse
import torch
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
import math

from .utils import (
    show_mask_on_image,
    preprocess_prompt,
    preprocess_image,
    get_heatmap,
    make_square
)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        
        ######
        category = line["text"].split(" ")[2:4]
        qs = f"Does there exist {category[0]} {category[1]} in the image? Answer the question using a single word or phrase."
        ######
        
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        prompt = prompt.replace(
            "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
            ""
        )

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image = make_square(image)
        image_tensor, images = process_images([image], self.image_processor, self.model_config)
        image = images[0]        

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)    
    image_tensors = torch.stack(image_tensors, dim=0)    
    return input_ids, image[0], image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)    
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    model.config.mm_patch_merge_type = "spatial"

    # resized_image_size = args.resized_image_size
    # positional_embedding_type = args.positional_embedding_type
    # image_resize_type = args.image_resize_type
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # if positional_embedding_type == "default":
    #     assert resized_image_size==336, "default embedding only allows size of 336"
    
    # else:
    #     print(f"change positional embedding to {positional_embedding_type}")
    #     # Default configurations of model position embedding
    #     patch_size = 14
    #     num_patches = (resized_image_size // patch_size) ** 2
    #     num_positions = num_patches + 1
    #     embed_dim = model.model.vision_tower.vision_tower.vision_model.embeddings.embed_dim

    #     model.model.vision_tower.vision_tower.vision_model.embeddings.image_size = resized_image_size
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.num_patches = num_patches
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.num_positions = num_positions
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)

    #     # Modify positional embedding to match the resized image size
    #     if positional_embedding_type == "zero":       
    #         model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(num_positions, embed_dim).to(device)
    #         nn.init.constant_(model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight, 0)
    #     elif positional_embedding_type == "interpolation":
    #         # Interpolate from the pretrained positional embedding
    #         original_embedding = model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data
    #         original_num_positions = original_embedding.size(0)
    #         new_embedding = torch.nn.functional.interpolate(
    #             original_embedding.unsqueeze(0).transpose(1, 2), 
    #             size=(num_positions,), 
    #             mode='linear', 
    #             align_corners=False
    #         ).transpose(1, 2).squeeze(0)
    #         model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(num_positions, embed_dim).to(device)
    #         model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
    #     elif positional_embedding_type == "reduced":
    #         # Reduce the pretrained embedding by truncating
    #         original_embedding = model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data
    #         model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(num_positions, embed_dim).to(device)
    #         model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(original_embedding[:num_positions])
                
    #     model.to(device)   
    
    # if image_resize_type == "modify_processor":
    #     image_processor.crop_size = {'height': resized_image_size, 'width': resized_image_size}
    #     image_processor.size = {'shortest_edge': resized_image_size}
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    try:
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
            args.conv_mode = args.conv_mode + '_mmtag'
            print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

        data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

        cnt = 0
        for (input_ids, image, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
            idx = line["question_id"]
            cur_prompt = line["text"]

            input_ids = input_ids.to(device='cuda', non_blocking=True)

            # if image_resize_type == "interpolate":
            #     image_tensor = F.interpolate(image_tensor, size=(resized_image_size, resized_image_size), mode='bilinear', align_corners=False)           
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=False,
                    # do_sample=True if args.temperature > 0 else False,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    # num_beams=args.num_beams,
                    # max_new_tokens=args.max_new_tokens,
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    image_mask=None)

            # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if output_ids["sequences"][0][0] == 1:
                output_ids["sequences"] = output_ids["sequences"][0][1:].unsqueeze(0)
            outputs = tokenizer.decode(output_ids["sequences"][0]).strip()
            # print(outputs)
            #print(output_ids["sequences"])        

            folder = f"/home/aidas_intern_1/woohyeon/Test_feasibility/vis/{str(idx).zfill(6)}"
            os.makedirs(folder, exist_ok=True)
            cnt += 1
            if args.recursion_type == "all":
                # print("recursion")
                heat_torch_stack, img_with_attn, ret_attn = get_heatmap(model, output_ids, tokenizer, cur_prompt, image, input_ids, folder)
                outputs = output_ids["sequences"]
                del output_ids

                med = torch.stack(heat_torch_stack, dim=0)
                med = med.mean(dim=0)
                np_img = np.array(image)[:, :, ::-1]
                for i, attn in enumerate(heat_torch_stack):
                    attn -= med
                    attn = torch.relu(attn)
                    attn = attn / attn.max()
                    img_with_attn, heatmap = show_mask_on_image(np_img, attn.numpy())
                    img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
                    tt = tokenizer.decode(outputs[0][i])
                    if tt == '</s>':
                        tt = "eos"
                    cv2.imwrite(f"{folder}/{str(i).zfill(3)}_{tt}.png", img_with_attn)
                
                med = torch.stack(ret_attn, dim=0)
                med = med.mean(dim=0)
                attn = ret_attn[0] - med
                attn = torch.relu(attn)
                attn = attn / attn.max()

                image_mask_list = []
                for row in range(attn.shape[0]):
                    for col in range(attn.shape[1]):
                        if attn[row, col] > args.recursion_threshold:
                            image_mask_list.append(torch.LongTensor([[row, col]]))
                image_mask = torch.cat(image_mask_list)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        image_sizes=image_sizes,
                        do_sample=False,
                        # do_sample=True if args.temperature > 0 else False,
                        # temperature=args.temperature,
                        # top_p=args.top_p,
                        # num_beams=args.num_beams,
                        # max_new_tokens=args.max_new_tokens,
                        max_new_tokens=512,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_attentions=True,
                        image_mask=image_mask)
                
                
                outputs = tokenizer.decode(output_ids["sequences"][0], skip_special_tokens=True).strip()
            
            # print(outputs)

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "answer_id": ans_id,
                                        "model_id": model_name,
                                        "metadata": {}}) + "\n")
            # ans_file.flush()
    except Exception as e:
        print(e)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    # parser.add_argument("--resized_image_size", type=int, default=336)
    # parser.add_argument("--image_resize_type", type=str, default="interpolate")
    # parser.add_argument("--positional_embedding_type", type=str, default="default")
    parser.add_argument("--recursion_type", type=str, default="all")
    parser.add_argument("--recursion_threshold", type=float, default=0.5)
    parser.add_argument("--save_image_attn_map", type=bool, default=False)
    parser.add_argument("--image_save_path", type=str, default="./playground/data/eval/pope/val_2014_attn_img")
    args = parser.parse_args()

    eval_model(args)
