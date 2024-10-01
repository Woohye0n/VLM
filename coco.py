import cv2
import numpy as np
from pycocotools import mask as maskUtils
from torchvision.datasets import CocoDetection
import random
 
# 데이터셋 경로 설정
data_dir = './val2017'
Caption_file = './annotations/captions_val2017.json' #Caption
instances_file = './annotations/instances_val2017.json' #Segmentation, category_id, bbox
keypoints_file = './annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox
 
# 데이터셋 로드
dataset = CocoDetection(root=data_dir, annFile=instances_file)
coco = dataset.coco
 
# 색상 목록
colors = []
for _ in range(50):
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    colors.append(color)
 
# 이미지와 인스턴스 정보 가져오기
for i in range(len(dataset)):
    image, target = dataset[i]
 
    # 이미지 시각화
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # 인스턴스 정보 가져오기
    for obj in target:
        segmentation = obj["segmentation"]
        bbox = obj["bbox"]
        category_id = obj["category_id"]
 
        color = colors[category_id % len(colors)]
        color_bgr = color[::-1]  # Convert RGB to BGR
 
        # 바운딩 박스 시각화
        bbox = [int(coord) for coord in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color_bgr, 2)
 
        #segmentation이 유효한지 확인
        if isinstance(segmentation, list):
            h, w = image.shape[:2]
 
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in segmentation:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255) # color is not specified here
 
            mask_color = np.stack([mask]*3, axis=-1)
            for i in range(3):
                mask_color[..., i][mask_color[..., i] == 255] = color[i]  # Use RGB color for mask
 
            alpha = ((mask_color > 0).max(axis=2) * 128).astype(np.uint8)
            rgba_mask = np.concatenate([mask_color, alpha[:, :, np.newaxis]], axis=2)
 
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            image_rgba = cv2.addWeighted(image_rgba, 1, rgba_mask, 0.5, 0)
 
            image = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)
 
        # 클래스 이름 가져오기
        cat_id = coco.getCatIds(catIds=[category_id])
        cat_name = coco.loadCats(cat_id)[0]['name']
 
        # 클래스 이름 시각화
        cv2.putText(image, cat_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
 
 
 
    # 이미지 출력
    cv2.imshow('Image', image)
    if cv2.waitKey(0) == ord('q'):
        break
 
cv2.destroyAllWindows()