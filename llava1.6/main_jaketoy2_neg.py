import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import get_model_name_from_path
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms
from torchmetrics import StructuralSimilarityIndexMeasure

from utils import (
    show_mask_on_image,
    preprocess_prompt,
    preprocess_image,
    get_heatmap,
    make_square
)

from torchvision.datasets import CocoDetection
import random
import monai

def calculate_entropy(sequence, scores):
    log_prob_sum = 0.0  # Log-probability sum for calculating entropy
    entropy_sum = 0.0  # Sum of entropies
    cumulative_confidences = [1]

    for idx, token_id in enumerate(sequence): 
        if idx != 0: 
            probs = F.softmax(scores[idx-1], dim=-1)
            token_prob = probs[0, token_id].item()

            log_prob_sum += np.log(token_prob)  
            cumulative_confidences.append(np.exp(log_prob_sum))
            entropy_sum -= token_prob * np.log(token_prob)

    P_T_given_I_Q_Full = cumulative_confidences[-1] if cumulative_confidences else np.exp(log_prob_sum)

    # 확률 감소량 계산
    confidence_drops = [cumulative_confidences[i] / cumulative_confidences[i+1] for i in range(0, len(cumulative_confidences)-1)]

    # 가장 큰 감소를 유발한 인덱스 찾기
    max_drop_index = confidence_drops.index(max(confidence_drops)) + 1 # idx 보정

    return cumulative_confidences, P_T_given_I_Q_Full, entropy_sum, max_drop_index

def calculate_entropy2(sequence, scores):
    log_prob_sum_full = 0.0  # Log-probability sum for calculating full sequence probability
    entropy_sum = 0.0  # Sum of entropies
    probabilities = []  # List to store probabilities at specific indices
    for idx in range(0, len(sequence)): 
        logits = output_scores
        probs = F.softmax(logits[:, -1, :], dim=-1)
        token_prob = probs[0, sequence[idx]].item()  
        log_prob_sum_full += np.log(token_prob + 1e-10)
        
        # Store the cumulative probability up to the specified compare_index
        if idx == compare_index or idx == compare_index + 1 or idx == compare_index + 2:
            probabilities.append(np.exp(log_prob_sum_full))
        
        # Calculate entropy
        entropy_sum -= token_prob * np.log(token_prob + 1e-10)

    P_T_given_I_Q_1 = probabilities[0]  # Probability up to index compare_index
    P_T_given_I_Q_2 = probabilities[1]  # Probability up to index compare_index + 1
    P_T_given_I_Q_3 = probabilities[2]

    return P_T_given_I_Q_1, P_T_given_I_Q_2, P_T_given_I_Q_3
def run():
    # ===> specify the model path
    model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, # model_base
        model_name, 
        device_map=device
    )
    
    if model_path == "liuhaotian/llava-v1.6-vicuna-7b":
        model.config.mm_patch_merge_type = "spatial"
    # print(model)
    
    # Dataset path
    data_dir = './val2017'
    Caption_file = './annotations/captions_val2017.json' #Caption
    instances_file = './annotations/instances_val2017.json' #Segmentation, category_id, bbox
    keypoints_file = './annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox
    
    save = "/home/aidas_intern_1/woohyeon/llava1.6/jake_result2/Posibility_test_with_top20%_Long.csv"
    # Log files
    with open(f"./minmax.csv", "w") as f:
        f.write(f"hallucination, attn_max, attn_min\n")
    with open(f"./loss.csv", "w") as f:
        f.write(f"img_idx,cat_idx,hallucination,is_positive,cat_name,dice_ce,focal,l2,scaled_dice_ce,scaled_focal,scaled_l2\n")
    with open(f"{save}", "w") as f:
        f.write("hallucination1, hallucination2, Success, P_T_given_I_Q_Full1R, P_T_given_I_Q_Full2R \n")  # 헤더 추가

    # Load dataset
    dataset = CocoDetection(root=data_dir, annFile=instances_file)
    coco = dataset.coco

    # Categories
    categories = coco.loadCats(coco.getCatIds())
    categories_list = [cat['name'] for cat in categories]

    cnt = 0
    for img_idx, i in enumerate(range(len(dataset))):
        image, target = dataset[i]
        image = make_square(image)
        if image.height != image.width or image.height < 336 or image.width < 336:
            print("skipping")
            continue
        print("running")
        
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
        # k = len(positive_list)
        
        predict_list = []
        with open(f"./non/{str(cnt).zfill(5)}.txt", "r") as f:
            for l in f.readlines():
                predict_list.append(l.rstrip("\n"))
        # predict_list += random.sample(negative_list, 10)
        for cat_idx, cat_name in enumerate(predict_list):
            mask = mask_dict.get(cat_name, np.zeros((image.height, image.width), dtype=np.uint8))

            prompt_text = f"Does there exist {cat_name} in the image?"
            input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
            # print(f"\n{prompt}\n")
            
            images = [image]
            width, height = image.size
            image, image_tensor = preprocess_image(model, image_processor, images)
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
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    image_mask=None,
                    output_scores=True,
                )

            text = tokenizer.decode(outputs["sequences"][0]).strip()
            print(text)


            cumulative_confidences , P_T_given_I_Q_Full, entropy_sum, max_drop_index = calculate_entropy(sequence = outputs["sequences"][0],
                                                        scores=outputs.scores)

            # print(f'Cumulative_Confidences:  {cumulative_confidences}')
            # print(f'P_T_given_I_Q_Full (P(T|I,Q)):  {P_T_given_I_Q_Full}')
            # print(f'Entropy_sum:  {entropy_sum}')
            print(f'Index of Maximum Probability Drop: {tokenizer.decode(outputs["sequences"][0][max_drop_index]).strip()}')

            answer = ""
            if "Yes" in text:
                answer = "yes"
            elif "No" in text:
                answer = "no"
            
            is_positive = cat_idx < k
            # hallucination = (is_positive and answer == "no") or (not is_positive and answer == "yes")
            hallucination1 = answer == "yes"
            heat_torch_stack, img_with_attn, ret_attn = get_heatmap(model, outputs, tokenizer, prompt, image, input_ids)
            del outputs
            # with open(f"/home/aidas_intern_1/woohyeon/llava1.6/jake_result/Token_P_compare.csv", "a") as f:
            #     f.write(f"{hallucination}, {P_T_given_I_Q1}, {P_T_given_I_Q2}, {P_T_given_I_Q3}\n")

            ###################### 2nd Stage ##########################
            cumulative_confidences2 = None 
            P_T_given_I_Q_Full2 = None
            entropy_sum2 = None

            if P_T_given_I_Q_Full < 0.75:

                # print(f'2nd Stage Starts!')
                top_k_percent = 0.2
                
                attn = ret_attn[max_drop_index]

                ## Setting Threshold (Top 20%)
                flattened_attn = attn.view(-1) 
                flattened_attn = flattened_attn.float()
                threshold_index = int(len(flattened_attn) * (1 - top_k_percent)) 
                threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

                image_mask_list = []
                for row in range(attn.shape[0]):
                    for col in range(attn.shape[1]):
                        if attn[row, col] > threshold_value:
                            image_mask_list.append(torch.LongTensor([[row, col]]))
                image_mask = torch.cat(image_mask_list)
                
                # print(image_mask)
                prompt_text = f"Does there exist {cat_name} in the image?"
                input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
                # print(f"\n{prompt}\n")

                ########################
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
                        image_mask=image_mask,
                        output_scores=True,
                        
                    )

                # print(outputs["sequences"][0])
                text = tokenizer.decode(outputs["sequences"][0]).strip()
                # print(text)
                # print(outputs["attentions"][0].shape)
                
                cumulative_confidences2, P_T_given_I_Q_Full2, entropy_sum2,_ = calculate_entropy(sequence = outputs["sequences"][0],
                                                        scores=outputs.scores)
                answer = ""
                if "Yes" in text:
                    answer = "yes"
                elif "No" in text:
                    answer = "no"
                
                is_positive = cat_idx < k
                # hallucination = (is_positive and answer == "no") or (not is_positive and answer == "yes")
                hallucination2 = answer == "yes"
            
            else:
                hallucination2 = None
                sucess =  None
                
            success = hallucination1 and (hallucination2 == False) if hallucination2 is not None else None
            with open(f"{save}", "a") as f:
                f.write(f"{hallucination1},{hallucination2},{success},{P_T_given_I_Q_Full}, {P_T_given_I_Q_Full2} \n")

            continue
            ################################
            
            print(len(heat_torch_stack))
            print(heat_torch_stack[0].shape)
            
            os.makedirs(f"./img/{img_idx}_{cat_idx}", exist_ok=True)
            np_img = np.array(image)[:, :, ::-1]
            cv2.imwrite(f"./img/{img_idx}_{cat_idx}/origin.png", np_img)
            cv2.imwrite(f"./img/{img_idx}_{cat_idx}/origin_attn.png", img_with_attn)
            
            # if answer == "yes":
            #     start_idx = 5
            # elif answer == "no":
            #     start_idx = 6
            # else:
            #     bp = 'bp'
            start_idx = 0

            med = torch.stack(heat_torch_stack, dim=0)
            med = med.mean(dim=0)
            avg_attn = torch.zeros_like(heat_torch_stack[start_idx])
            cnt = 0
            for i, attn in enumerate(heat_torch_stack):
                cnt += 1
                
                attn -= med
                avg_attn += attn      
                attn = torch.relu(attn)
                attn = attn / attn.max()
                img_with_attn, heatmap = show_mask_on_image(np_img, attn.numpy())
                img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
                # tt = tokenizer.decode(outputs["sequences"][0][start_idx + 1 + i], add_special_tokens=False).strip()
                cv2.imwrite(f"./img/{img_idx}_{cat_idx}/{hallucination}_token_{i}.png", img_with_attn)
            
            scaled_attn = avg_attn / cnt
            scaled_attn /= scaled_attn.max()
            scaled_attn -= 0.3
            scaled_attn = (scaled_attn > 0).float()

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
            dice_loss = monai.losses.DiceLoss()

            dice_ce = dice_ce_loss(gt, attn_unsq)
            focal = focal_loss(gt, attn_unsq)
            l2 = l2_loss(gt, attn_unsq)
            
            scaled_dice_ce = dice_ce_loss(gt, scaled_attn_unsq)
            scaled_focal = focal_loss(gt, scaled_attn_unsq)
            scaled_l2 = l2_loss(gt, scaled_attn_unsq)
            scaled_dice = dice_loss(gt,attn_unsq)

            print(dice_ce.item(), focal.item(), l2.item(), scaled_dice_ce.item(), scaled_focal.item(), scaled_l2.item())
            
            # with open(f"./loss_jake.csv", "a") as f:
            #     f.write(f"{img_idx},{cat_idx},{hallucination},{is_positive},{cat_name},{scaled_dice_ce.item()},{scaled_focal.item()},{scaled_l2.item()}\n")
            
            with open(f"/home/aidas_intern_1/woohyeon/llava1.6/jake_result/PTI_Cos_sim_TEST8.csv", "a") as f:
                f.write(f"{hallucination}, {P_T_given_I_Q1}, {scaled_dice}\n")
            
                       
if __name__ == '__main__':
    run()