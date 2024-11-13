import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import cv2
import copy

import torch
import torch.nn.functional as F

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from utils import (
    show_mask_on_image,
    preprocess_prompt,
    preprocess_image,
    get_heatmap,
)

from torchvision.datasets import CocoDetection
import random
import monai

def run():
    # ===> specify the model path
    model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"
    model_path = "lmms-lab/llava-onevision-qwen2-7b-si"

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, # model_base
        "llava_llama3", 
        device_map=device,
        attn_implementation=None
    )
    
    model.eval()
    model.tie_weights()

    # Dataset path
    data_dir = './val2017'
    Caption_file = './annotations/captions_val2017.json' #Caption
    instances_file = './annotations/instances_val2017.json' #Segmentation, category_id, bbox
    keypoints_file = './annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox
    
    # Log files
    with open(f"./minmax.csv", "w") as f:
        f.write(f"hallucination, attn_max, attn_min\n")
    with open(f"./loss.csv", "w") as f:
        f.write(f"img_idx,cat_idx,hallucination,is_positive,cat_name,dice_ce,focal,l2,scaled_dice_ce,scaled_focal,scaled_l2\n")
    
    # Load dataset
    dataset = CocoDetection(root=data_dir, annFile=instances_file)
    coco = dataset.coco

    # Categories
    categories = coco.loadCats(coco.getCatIds())
    categories_list = [cat['name'] for cat in categories]

    for img_idx, i in enumerate(range(len(dataset))):
        image, target = dataset[i]
        
        # Create semantic mask from instance mask.
        mask_dict = {}
        for obj in target:
            # Get category name
            category_id = coco.getCatIds(catIds=[obj["category_id"]])
            name = coco.loadCats(category_id)[0]['name']
            
            # Get instance mask
            segmentation = obj["segmentation"]
            if type(segmentation) != list:
                continue
            mask = np.zeros_like(image, dtype=np.uint8)
            poly = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], (255, 255, 255))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Merge to semantic mask
            semantic_mask = mask_dict.get(name)
            if semantic_mask is not None:
                semantic_mask = semantic_mask | mask
            else:
                semantic_mask = mask
            
            mask_dict[name] = semantic_mask
        
        # Create sorted list by mask area
        positive_list = sorted(mask_dict, key=lambda x: mask_dict[x].sum(), reverse=True)
        # Create non-existent objects list
        negative_list = [name for name in categories_list if name not in positive_list]
        
        # Initialize visualize image
        vis_image = np.array(image)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        # Predict top-k-area positives & k-random negatives
        k = 0
        if len(positive_list) > 3:
            k = 3
        else:
            k = len(positive_list)
        
        predict_list = positive_list[:k]
        predict_list += random.sample(negative_list, k)
        for cat_idx, cat_name in enumerate(predict_list):
            mask = mask_dict.get(cat_name, np.zeros((image.height, image.width), dtype=np.uint8))

            conv_template = "llava_llama_3"
            prompt_text = f"Does there exist {cat_name} in the image? Answer in only 'yes' or 'no'."
            print(f"\n{prompt_text}\n")
            prompt_text = DEFAULT_IMAGE_TOKEN + prompt_text
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            print(prompt_question)
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            # input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
            images = [image]
            width, height = image.size
            # image, image_tensor = preprocess_image(model, image_processor, images)
            image_tensor = preprocess_image(model, image_processor, images)
            image_size = image.size
            ################################################
            ids_list = input_ids.tolist()[0]
            ids_list.append(2)
            input_ids_temp = torch.tensor(ids_list)
            # display(image)
            # print(prompt_text)

            # print(input_ids.shape)
            # generate the response
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )

            # print(outputs["sequences"])
            # text = tokenizer.decode(outputs["sequences"][0]).strip()
            # print(model)
            text = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
            print(outputs["sequences"])
            print(text)
            print(outputs["attentions"][2][0].shape)
            exit()
                
            answer = ""
            if "Yes" in text:
                answer = "yes"
            elif "No" in text:
                answer = "no"
            
            is_positive = cat_idx < k
            hallucination = (is_positive and answer == "no") or (not is_positive and answer == "yes")
            
            heat_torch_stack, img_with_attn = get_heatmap(model, outputs, tokenizer, prompt, image, input_ids)
            
            os.makedirs(f"./img/{img_idx}_{cat_idx}", exist_ok=True)
            np_img = np.array(image)[:, :, ::-1]
            cv2.imwrite(f"./img/{img_idx}_{cat_idx}/origin.png", np_img)
            
            if answer == "yes":
                start_idx = 5
            elif answer == "no":
                start_idx = 6
            else:
                bp = 'bp'

            med = torch.stack(heat_torch_stack, dim=0)
            med = med.mean(dim=0)
            avg_attn = torch.zeros_like(heat_torch_stack[start_idx])
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
                cv2.imwrite(f"./img/{img_idx}_{cat_idx}/{str(start_idx + i).zfill(4)}_token_{tt}.png", img_with_attn)
            
            attn = avg_attn
            scaled_attn = torch.relu(attn)
            # scaled_attn = scaled_attn * 10
            scaled_attn /= scaled_attn.max()

            img_with_attn, heatmap = show_mask_on_image(np_img, scaled_attn.numpy())
            img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
            # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
            # tt = tokenizer.decode(input_ids_ret[i], add_special_tokens=False).strip()
            cv2.imwrite(f"./img/{img_idx}_{cat_idx}/{str(i).zfill(4)}_{cat_name}.png", img_with_attn)
            # if tt == cat_name:
            with open(f"./minmax.csv", "a") as f:
                f.write(f"{hallucination}, {attn.max()}, {attn.min()}\n")

            h, w = mask.shape
            attn_h, attn_w = attn.size()
            mask_resize = mask
            
            # Resize mask to square size
            if h != attn_h:
                half_h = (attn_h - h) // 2
                block1 = np.zeros((half_h, w), dtype=np.uint8)
                block2 = np.zeros((attn_h - h - half_h, w), dtype=np.uint8)
                mask_resize = cv2.vconcat([block1, mask_resize, block2])
            if w != attn_w:
                half_w = (attn_w - w) // 2
                block1 = np.zeros((h, half_w), dtype=np.uint8)
                block2 = np.zeros((h, attn_w - w - half_w), dtype=np.uint8)
                mask_resize = cv2.hconcat([block1, mask_resize, block2])
            
            gt = torch.tensor(mask_resize, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gt /= gt.max()
            attn_unsq = attn.unsqueeze(0).unsqueeze(0)
            scaled_attn_unsq = scaled_attn.unsqueeze(0).unsqueeze(0)
            
            dice_ce_loss = monai.losses.DiceCELoss(sigmoid=False, squared_pred=True, reduction='mean')
            l2_loss = torch.nn.MSELoss()
            focal_loss = monai.losses.FocalLoss()
            
            dice_ce = dice_ce_loss(gt, attn_unsq)
            focal = focal_loss(gt, attn_unsq)
            l2 = l2_loss(gt, attn_unsq)
            
            scaled_dice_ce = dice_ce_loss(gt, scaled_attn_unsq)
            scaled_focal = focal_loss(gt, scaled_attn_unsq)
            scaled_l2 = l2_loss(gt, scaled_attn_unsq)
            print(dice_ce.item(), focal.item(), l2.item(), scaled_dice_ce.item(), scaled_focal.item(), scaled_l2.item())
            
            with open(f"./loss.csv", "a") as f:
                f.write(f"{img_idx},{cat_idx},{hallucination},{is_positive},{cat_name},{scaled_dice_ce.item()},{scaled_focal.item()},{scaled_l2.item()}\n")
            
if __name__ == '__main__':
    run()