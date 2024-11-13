# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea
from io import BytesIO
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ..conversation import conv_templates
from ..mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def preprocess_prompt(model, model_name, prompt_text, tokenizer):
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
    return input_ids, prompt


def preprocess_image(model, image_processor, images):
    image_tensor, images = process_images(images, image_processor, model.config)
    # image_tensor = F.interpolate(image_tensor, size=(118, 118), mode='bilinear', align_corners=False)
    image = images[0]
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    return image, image_tensor


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


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def get_heatmap(model, outputs, tokenizer, prompt, image, input_ids, folder):
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
        
    # plot the trend of attention weights over the vision tokens
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(overall_attn_weights_over_vis_tokens)
    ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
    ax.set_xticklabels(
        [tokenizer.decode(token, add_special_tokens=False).strip() for token in outputs["sequences"][0].tolist()],
        rotation=75
    )
    ax.set_title("at each token, the sum of attention weights over all the vision tokens")
    plt.savefig(folder + "/plt.png")

    # Connect with the vision encoder attention
    # to visualize the attention over the image.
    # vis_attn_matrix will be of torch.Size([N, N])
    # where N is the number of vision tokens/patches
    # `all_prev_layers=True` will average attention from all layers until the selected layer
    # otherwise only the selected layer's attention will be used
    # print(len(model.get_vision_tower().image_attentions), model.get_vision_tower().select_layer)
    # print(model.get_vision_tower().image_attentions[0].shape)
    image_attentions = []
    for layer in model.get_vision_tower().image_attentions:
        layer = layer[0, ...].unsqueeze(0)
        image_attentions.append(layer)
        # print(layer.shape)
    # vis_attn_matrix = aggregate_vit_attention(
    #     model.get_vision_tower().image_attentions,
    #     select_layer=model.get_vision_tower().select_layer,
    #     all_prev_layers=True
    # )
    vis_attn_matrix = aggregate_vit_attention(
        image_attentions,
        select_layer=model.get_vision_tower().select_layer,
        all_prev_layers=True
    )
    # print(vis_attn_matrix.shape)
    grid_size = model.get_vision_tower().num_patches_per_side

    # whether visualize the attention heatmap or 
    # the image with the attention heatmap overlayed

    output_token_inds = list(range(output_token_start, output_token_end))
    heat_torch_stack = []
    
    ####
    #### input / ouput swap 가능
    ####
    ## output
    np_img = np.array(image)[:, :, ::-1]
    ret_attn = []
    for i in range(len(output_token_inds)):
    ## input
    # for i, ax in enumerate(input_ids[0]):

        # target_token_ind = i
        target_token_ind = output_token_inds[i]
        attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

        attn_over_image = []
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
            vis_attn = vis_attn.reshape(grid_size, grid_size)
            # vis_attn = vis_attn / vis_attn.max()
            # attn_over_image.append(vis_attn)
            attn_over_image.append(vis_attn * weight)
        attn_over_image = torch.stack(attn_over_image).sum(dim=0)
        # print("max: ", attn_over_image.max(), "min :", attn_over_image.min())
        attn_over_image = attn_over_image / attn_over_image.max()
        ret_attn.append(attn_over_image)
        attn_over_image = attn_over_image.to(torch.float32)

        attn_over_image = F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0), 
            size=image.size, 
            # mode='nearest', 
            mode='bicubic',
            align_corners=True
        ).squeeze()
        heat_torch_stack.append(attn_over_image)

    attn_tot = sum(heat_torch_stack)
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_tot.numpy())
    # tt = tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
    # tt = tokenizer.decode(input_ids[0][i], add_special_tokens=False).strip()
    img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)

    return heat_torch_stack, img_with_attn, ret_attn

def make_square(im, min_size=200, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im