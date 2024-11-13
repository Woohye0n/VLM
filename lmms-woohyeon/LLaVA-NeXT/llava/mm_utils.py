from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

## Additional imports for utils
import requests
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def resize_and_center_crop(image, shortest_edge_length):
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image, grid_params):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image, data_args, processor=None):
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    divide_shape = (height // patch_size, width // patch_size) # divide_ shape for recursion
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    # return patches
    return patches, divide_shape


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    # patches = divide_to_patches(image_padded, processor.crop_size["height"])
    patches, divide_shape = divide_to_patches(image_padded, processor.crop_size["height"]) # divide_shape = (H, W)

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)

    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    # return torch.stack(image_patches, dim=0)
    return torch.stack(image_patches, dim=0), divide_shape


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None) # anyres_max_9 for onevision
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image, divide_shape = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images, divide_shape


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea

def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap

def get_heatmap(
    vision_tower,
    outputs,
    tokenizer,
    prompt,
    image,
    input_ids,
    base_token_replace=False,
    divide_shape=None,
    plt_save_folder=None):
    
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
    input_token_len = vision_tower.vision_tower.vision_model.embeddings.num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + vision_tower.vision_tower.vision_model.embeddings.num_patches
    output_token_len = len(outputs["sequences"][0])
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len
    
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
        
    if plt_save_folder is not None:
        # plot the trend of attention weights over the vision tokens
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(overall_attn_weights_over_vis_tokens)
        ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
        ax.set_xticklabels(
            [tokenizer.decode(token, add_special_tokens=False).strip() for token in outputs["sequences"][0].tolist()],
            rotation=75
        )
        ax.set_title("at each token, the sum of attention weights over all the vision tokens")
        plt.savefig(plt_save_folder + "/plt.png")

    # Connect with the vision encoder attention
    # to visualize the attention over the image.
    # vis_attn_matrix will be of torch.Size([N, N])
    # where N is the number of vision tokens/patches
    # `all_prev_layers=True` will average attention from all layers until the selected layer
    # otherwise only the selected layer's attention will be used
    
    image_attentions = []
    for layer in vision_tower.image_attentions:
        layer = layer[0, ...].unsqueeze(0)
        image_attentions.append(layer)
        
    vis_attn_matrix = aggregate_vit_attention(
        image_attentions,
        select_layer=vision_tower.select_layer,
        all_prev_layers=True
    )
    grid_size = int(math.sqrt(vision_tower.vision_tower.vision_model.embeddings.num_patches))

    # whether visualize the attention heatmap or 
    # the image with the attention heatmap overlayed

    output_token_inds = list(range(output_token_start, output_token_end))
    # print(f"input_token_len: {input_token_len}")
    # print(f"input_ids_shape: {input_ids.shape}")
    # print(f"llm_attn_size: {llm_attn_matrix.shape}")
    # print(f"output_token_inds: {output_token_inds}")
    heat_torch_stack = []
    ret_attn = []
    
    ####
    ## output
    for i in range(len(output_token_inds)):    
        target_token_ind = output_token_inds[i]
        #print(f"target_token_indices: {target_token_ind}")        
        attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

        attn_over_image = []
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
            vis_attn = vis_attn.reshape(grid_size, grid_size)            
            # vis_attn = vis_attn / vis_attn.max()
            attn_over_image.append(vis_attn * weight)
            
        attn_over_image = torch.stack(attn_over_image).sum(dim=0)
        attn_over_image = attn_over_image / attn_over_image.max()
        #print(attn_over_image)
        ret_attn.append(attn_over_image)

        attn_over_image = F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0), 
            size=image.size, 
            # mode='nearest', 
            mode='bicubic', align_corners=True
        ).squeeze()
        heat_torch_stack.append(attn_over_image)

    # np_img = np.array(image)[:, :, ::-1]
    # attn_tot = sum(heat_torch_stack)
    # img_with_attn, heatmap = show_mask_on_image(np_img, attn_tot.numpy())
    # # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
    # # tt = tokenizer.decode(input_ids[0][i], add_special_tokens=False).strip()
    # img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)

    return heat_torch_stack, ret_attn

def make_square(im, min_size=200, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im