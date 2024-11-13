import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import cv2

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

def run():
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

    cnt = 0
    for img_idx, i in enumerate(range(len(dataset))):
        image, target = dataset[i]
        image = make_square(image)
        if image.height != image.width or image.height < 336 or image.width < 336:
            print("skipping")
            continue
        print(img_idx)
        
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
        k = len(positive_list)
        
        predict_list = []
        # predict_list = positive_list[:k]
        predict_list += random.sample(negative_list, 3)
        with open(f"./non/{str(cnt).zfill(5)}.txt", "w") as f:
            for c in predict_list:
                f.write(c + "\n")
        cnt += 1

if __name__ == '__main__':
    run()