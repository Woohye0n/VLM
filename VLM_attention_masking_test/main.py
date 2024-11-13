import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import get_model_name_from_path

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

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(\
        model_path, 
        None, # model_base
        model_name, 
        load_8bit, 
        load_4bit, 
        device=device
    )
    
    # Dataset path
    data_dir = '/home/aidas_intern_1/woohyeon/VLM/val2017'
    Caption_file = '/home/aidas_intern_1/woohyeon/VLM/annotations/captions_val2017.json' #Caption
    instances_file = '/home/aidas_intern_1/woohyeon/VLM/annotations/instances_val2017.json' #Segmentation, category_id, bbox
    keypoints_file = '/home/aidas_intern_1/woohyeon/VLM/annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox
    
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

    os.makedirs(f"/home/aidas_intern_1/woohyeon/VLM/masked", exist_ok=True)
    for img_idx, i in enumerate(range(len(dataset))):
        image, target = dataset[i]

        # Create semantic mask from instance mask.
        mask_dict = {}
        totalmask = np.zeros_like(image, dtype=np.uint8)
        totalmask = cv2.cvtColor(totalmask, cv2.COLOR_BGR2GRAY)
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
            totalmask = totalmask | semantic_mask
            
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
        ####
        k = len(positive_list)
            
        predict_list = positive_list[:k]
        # predict_list += random.sample(negative_list, k)
        for cat_idx, cat_name in enumerate(predict_list):
            mask = mask_dict.get(cat_name, np.zeros((image.height, image.width), dtype=np.uint8))

            prompt_text = f"Does there exist {cat_name} in the image? Answer in the format of 'Yes, there is {cat_name}.' or 'No, there is not {cat_name}.'"
            print(f"\n{prompt_text}\n")

            image_size = image.size
            origin_image = np.array(image)
            temp_image = np.array(image, dtype=np.float32)
            temp_image *= 0.1
            temp_image = cv2.GaussianBlur(temp_image, (3, 3), 1)
            temp_image = temp_image.astype(np.uint8)
            temp_image[mask!=0] = origin_image[mask!=0]

            rows, cols = np.nonzero(mask)
            col_min = cols.min()
            col_max = cols.max()
            row_min = rows.min()
            row_max = rows.max()
            w = col_max - col_min
            h = row_max - row_min
            x_min = max(0, col_min - (w * 2))
            x_max = min(image_size[0], col_max + (w * 2))
            y_min = max(0, row_min - (h * 2))
            y_max = min(image_size[1], row_max + (h * 2))
            temp_image = temp_image[y_min:y_max, x_min:x_max]

            temp_image = Image.fromarray(temp_image)

            input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
            resized_image, image_tensor = preprocess_image(model, image_processor, temp_image)
            image_size = resized_image.size
            ################################################
            ids_list = input_ids.tolist()[0]
            ids_list.append(2)
            input_ids_temp = torch.tensor(ids_list)
            # display(resized_image)
            # print(prompt_text)

            # generate the response
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    mask = mask,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )

            text = tokenizer.decode(outputs["sequences"][0]).strip()
            print(img_idx, text)
                        
            answer = ""
            if "Yes" in text:
                answer = "yes"
            elif "No" in text:
                answer = "no"
            
            is_positive = cat_idx < k
            hallucination = (is_positive and answer == "no") or (not is_positive and answer == "yes")

            if hallucination:
                with open(f"/home/aidas_intern_1/woohyeon/VLM/hallucination5.csv", "a") as f:
                    f.write(f"{img_idx},{cat_idx}\n")
                temp_image.save(f"/home/aidas_intern_1/woohyeon/VLM/masked5/{img_idx}_{cat_idx}_{cat_name}.png", "png")
                continue
                origin_image = np.array(image)
                temp_image = np.array(image, dtype=np.float32)
                temp_image *= 0.1
                temp_image = cv2.GaussianBlur(temp_image, (3, 3), 1)
                temp_image = temp_image.astype(np.uint8)
                temp_image[mask!=0] = origin_image[mask!=0]

                rows, cols = np.nonzero(mask)
                col_min = cols.min()
                col_max = cols.max()
                row_min = rows.min()
                row_max = rows.max()
                w = col_max - col_min
                h = row_max - row_min
                x_min = max(0, col_min - (w // 2))
                x_max = min(image_size[0], col_max + (w // 2))
                y_min = max(0, row_min - (h // 2))
                y_max = min(image_size[1], row_max + (h // 2))
                temp_image = temp_image[y_min:y_max, x_min:x_max]

                temp_image = Image.fromarray(temp_image)
                temp_image.save(f"/home/aidas_intern_1/woohyeon/VLM/masked3/{img_idx}_{cat_idx}.png", "png")
                resized_image2, image_tensor2 = preprocess_image(model, image_processor, temp_image)
                image_size2 = resized_image2.size
                with torch.inference_mode():
                    outputs2 = model.generate(
                        input_ids,
                        images=image_tensor2,
                        image_sizes=[image_size2],
                        do_sample=False,
                        max_new_tokens=512,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_attentions=True,
                    )

                    text2 = tokenizer.decode(outputs2["sequences"][0]).strip()
                print(text2)
                if "Yes" in text2:
                    with open(f"/home/aidas_intern_1/woohyeon/VLM/hallucination3.csv", "a") as f:
                        f.write(f"{img_idx},{cat_idx},{1}\n")
                elif "No" in text2:
                    with open(f"/home/aidas_intern_1/woohyeon/VLM/hallucination3.csv", "a") as f:
                        f.write(f"{img_idx},{cat_idx},{0}\n")
            continue

            
            heat_torch_stack, img_with_attn = get_heatmap(model, outputs, tokenizer, prompt, resized_image, input_ids)
            
            os.makedirs(f"/home/aidas_intern_1/woohyeon/VLM/img/{img_idx}_{cat_idx}", exist_ok=True)
            np_img = np.array(resized_image)[:, :, ::-1]
            cv2.imwrite(f"/home/aidas_intern_1/woohyeon/VLM/img/{img_idx}_{cat_idx}/origin.png", np_img)
            
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
                cv2.imwrite(f"/home/aidas_intern_1/woohyeon/VLM/img/{img_idx}_{cat_idx}/{str(start_idx + i).zfill(4)}_token_{tt}.png", img_with_attn)
            
            attn = avg_attn
            scaled_attn = torch.relu(attn)
            scaled_attn *= 10
            # attn = torch.relu(avg_attn)
            # attn = attn / attn.max()

            img_with_attn, heatmap = show_mask_on_image(np_img, scaled_attn.numpy())
            img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
            # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
            # tt = tokenizer.decode(input_ids_ret[i], add_special_tokens=False).strip()
            cv2.imwrite(f"/home/aidas_intern_1/woohyeon/VLM/img/{img_idx}_{cat_idx}/{str(i).zfill(4)}_{cat_name}.png", img_with_attn)
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
            if gt.max() != 0:
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
            
            with open(f"/home/aidas_intern_1/woohyeon/VLM/loss.csv", "a") as f:
                f.write(f"{img_idx},{cat_idx},{hallucination},{is_positive},{cat_name},{dice_ce.item()},{focal.item()},{l2.item()},{scaled_dice_ce.item()},{scaled_focal.item()},{scaled_l2.item()}\n")
                
if __name__ == '__main__':
    run()