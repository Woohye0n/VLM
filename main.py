import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from utils import (
    load_image, 
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

from pycocotools import mask as maskUtils
import torchvision
from torchvision.datasets import CocoDetection
import random
import monai

# ===> specify the model path
model_path = "liuhaotian/llava-v1.5-7b"

# load the model
load_8bit = False
load_4bit = False
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None, # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)
 
# Dataset path
data_dir = './val2017'
Caption_file = './annotations/captions_val2017.json' #Caption
instances_file = './annotations/instances_val2017.json' #Segmentation, category_id, bbox
keypoints_file = './annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox
 
# Load dataset
dataset = CocoDetection(root=data_dir, annFile=instances_file)
coco = dataset.coco

for idx, i in enumerate(range(len(dataset))):
    image, target = dataset[i]
    
    # Create semantic mask from instance mask.
    mask_dict = {}
    for obj in target:
        # Get category name
        category_id = coco.getCatIds(catIds=[obj["category_id"]])
        category_name = coco.loadCats(category_id)[0]['name']
        
        # Get instance mask
        segmentation = obj["segmentation"]
        mask = np.zeros_like(image, dtype=np.uint8)
        poly = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [poly], (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Merge to semantic mask
        category_mask = mask_dict.get(category_name)
        if category_mask is not None:
            category_mask = category_mask | mask
        else:
            category_mask = mask
        
        mask_dict[category_name] = category_mask
    
    # Create sorted list by mask area
    category_list = sorted(mask_dict, key=lambda x: mask_dict[x].sum(), reverse=True)
    
    # Initialize visualize image
    vis_image = np.array(image)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    # Predict top-k-area objects
    k = 2
    for idx2, cat_name in enumerate(category_list[:k]):
        mask = mask_dict[cat_name]

        prompt_text = f"Does there exist {cat_name} in the image? Answer in the format of 'Yes, there is {cat_name}.' or 'No, there is not {cat_name}.'"
        print(f"\n{prompt_text}\n")
        
        ################################################
        # preparation for the generation
        # unlikely that you need to change anything here
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        # image = load_image(image_path_or_url)
        image_tensor, images = process_images([image], image_processor, model.config)
        image = images[0]
        image_size = image.size
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # manually removing the system prompt here
        # otherwise most attention will be somehow put on the system prompt
        prompt = prompt.replace(
            "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
            ""
        )

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        ################################################
        ids_list = input_ids.tolist()[0]
        ids_list.append(2)
        input_ids_temp = torch.tensor(ids_list)
        # display(image)
        # print(prompt_text)

        # generate the response
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        text = tokenizer.decode(outputs["sequences"][0]).strip()
        print(text)
        
        ret = -1
        if "Yes" in text:
            ret = 1
        elif "No" in text:
            ret = 0

        # constructing the llm attention matrix
        aggregated_prompt_attention = []
        for i, layer in enumerate(outputs["attentions"][0]):
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            cur = attns_per_head[:-1].cpu().clone()
            # following the practice in `aggregate_llm_attention`
            # we are zeroing out the attention to the first <bos> token
            # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
            # we don't do this because <bos> is the only token that it can attend to
            cur[1:, 0] = 0.
            cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
            aggregated_prompt_attention.append(cur)
        aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

        # llm_attn_matrix will be of torch.Size([N, N])
        # where N is the total number of input (both image and text ones) + output tokens
        llm_attn_matrix = heterogenous_stack(
            [torch.tensor([1])]
            + list(aggregated_prompt_attention) 
            + list(map(aggregate_llm_attention, outputs["attentions"]))
        )

        # identify length or index of tokens
        input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
        vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
        vision_token_end = vision_token_start + model.get_vision_tower().num_patches
        output_token_len = len(outputs["sequences"][0])
        output_token_start = input_token_len
        output_token_end = input_token_len + output_token_len

        # look at the attention weights over the vision tokens
        overall_attn_weights_over_vis_tokens = []
        for i, (row, token) in enumerate(
            zip(
                llm_attn_matrix[input_token_len:], 
                outputs["sequences"][0].tolist()
            )
        ):
            overall_attn_weights_over_vis_tokens.append(
                row[vision_token_start:vision_token_end].sum().item()
            )

        # Connect with the vision encoder attention
        # to visualize the attention over the image.
        # vis_attn_matrix will be of torch.Size([N, N])
        # where N is the number of vision tokens/patches
        # `all_prev_layers=True` will average attention from all layers until the selected layer
        # otherwise only the selected layer's attention will be used
        vis_attn_matrix = aggregate_vit_attention(
            model.get_vision_tower().image_attentions,
            select_layer=model.get_vision_tower().select_layer,
            all_prev_layers=True
        )
        grid_size = model.get_vision_tower().num_patches_per_side

        # whether visualize the attention heatmap or 
        # the image with the attention heatmap overlayed

        output_token_inds = list(range(output_token_start, output_token_end))
        heat_torch_stack = []
        
        ####
        #### input / ouput swap 가능
        ####
        ## output
        for i in range(len(output_token_inds)):
        ## input
        # for i, ax in enumerate(input_ids[0]):

            # target_token_ind = i
            target_token_ind = output_token_inds[i] - 1
            attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
            attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

            attn_over_image = []
            for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
                vis_attn = vis_attn.reshape(grid_size, grid_size)
                # vis_attn = vis_attn / vis_attn.max()
                attn_over_image.append(vis_attn * weight)
            attn_over_image = torch.stack(attn_over_image).sum(dim=0)
            attn_over_image = attn_over_image / attn_over_image.max()

            attn_over_image = F.interpolate(
                attn_over_image.unsqueeze(0).unsqueeze(0), 
                size=image.size, 
                # mode='nearest', 
                mode='bicubic', align_corners=True
            ).squeeze()
            heat_torch_stack.append(attn_over_image)

            np_img = np.array(image)[:, :, ::-1]
            img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
            # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
            # tt = tokenizer.decode(input_ids[0][i], add_special_tokens=False).strip()
            img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)

        os.makedirs(f"./img/{ret}_{idx}_{idx2}", exist_ok=True)
        med = torch.stack(heat_torch_stack, dim=0)
        med = med.mean(dim=0)
        np_img = np.array(image)[:, :, ::-1]
        cv2.imwrite(f"./img/{ret}_{idx}_{idx2}/origin.png", np_img)
        
        if ret == 1:
            start_idx = 5
        elif ret == 0:
            start_idx = 6
        else:
            bp = 'bp'

        # avg_attn = heat_torch_stack[start_idx]
        avg_attn = torch.zeros_like(heat_torch_stack[start_idx])
        # avg_attn -= med
        
        cnt = 0
        for i, attn in enumerate(heat_torch_stack[start_idx : -2]):
            cnt += 1
            
            attn -= med
            avg_attn += attn
            
            ###
            ### Save pickle for SAM input
            ###
            # if i == 6:
            #     import pickle
            #     myd = {}
            #     temp = F.interpolate(attn.unsqueeze(0).unsqueeze(0),  (256, 256), mode="bilinear", align_corners=False).squeeze().numpy()
            #     temp = temp + 0.2 # bias  
            #     temp[temp > 1 - 0.001] = 1 - 0.001
            #     temp[temp < 0.001] = 0.001
            #     mask = (temp > 0.5).astype(np.uint8) * 255 # vis
            #     temp = np.log(temp / (1 - temp)) # inv_sigmoid
            #     myd["heat"] = temp
            #     with open('my_dict.pkl', 'wb') as f:
            #         pickle.dump(myd, f)
            
            attn = torch.relu(attn)
            attn = attn / attn.max()
            img_with_attn, heatmap = show_mask_on_image(np_img, attn.numpy())
            img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
            tt = tokenizer.decode(outputs["sequences"][0][start_idx + i], add_special_tokens=False).strip()
            cv2.imwrite(f"./img/{ret}_{idx}_{idx2}/{str(start_idx + i).zfill(4)}_tt_{tt}.png", img_with_attn)
        
        
        attn = torch.relu(avg_attn)
        attn = attn / attn.max()
        attn -= 0.3
        attn = torch.relu(attn)
        attn = attn / attn.max()

        img_with_attn, heatmap = show_mask_on_image(np_img, attn.numpy())
        img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
        # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
        # tt = tokenizer.decode(input_ids_ret[i], add_special_tokens=False).strip()
        cv2.imwrite(f"./img/{ret}_{idx}_{idx2}/{str(i).zfill(4)}_{cat_name}.png", img_with_attn)
        # if tt == cat_name:
        h, w = mask.shape
        attn_h, attn_w = attn.size()
        mask1 = mask
        
        if h != attn_h:
            half_h = (attn_h - h) // 2
            block1 = np.zeros((half_h, w), dtype=np.uint8)
            block2 = np.zeros((attn_h - h - half_h, w), dtype=np.uint8)
            mask1 = cv2.vconcat([block1, mask1, block2])
        if w != attn_w:
            half_w = (attn_w - w) // 2
            block1 = np.zeros((h, half_w), dtype=np.uint8)
            block2 = np.zeros((h, attn_w - w - half_w), dtype=np.uint8)
            mask1 = cv2.hconcat([block1, mask1, block2])
            
        gt = torch.tensor(mask1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        attn_unsq = attn.unsqueeze(0).unsqueeze(0)
        seg_loss = monai.losses.DiceCELoss(sigmoid=False, squared_pred=True, reduction='mean')
        l2_loss = torch.nn.MSELoss()
        loss = seg_loss(gt, attn_unsq)
        l2 = l2_loss(gt, attn_unsq)
        print(loss.item(), l2.item())
        
        with open(f"./log.csv", "a") as f:
            f.write(f"{ret},{idx},{idx2},{cat_name},{loss.item()},{l2.item()}\n")