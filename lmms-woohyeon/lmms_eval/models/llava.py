import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

import numpy as np
import torch.nn.functional as F
import csv
import os
import cv2
from pathlib import Path

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        show_mask_on_image,
        get_heatmap,
        make_square,
        process_anyres_image,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

def calculate_entropy_and_all_confidences(sequence, scores):
    """
    Calculate entropy, full sequence confidence, and per-token cumulative confidences.
    Args:
        sequence (torch.Tensor): Generated sequence of token IDs.
        scores (list of torch.Tensor): List of logit tensors, one for each token in the sequence.

    Returns:
        tuple: Full sequence confidence, entropy, and a list of per-token cumulative confidences.
    """
    log_prob_sum_full = 0.0  # Log-probability sum for calculating full sequence confidence
    entropy_sum = 0.0  # Sum of entropies
    cumulative_confidences = []  # List to store cumulative confidence up to each token

    for idx, token_id in enumerate(sequence):
        probs = F.softmax(scores[idx], dim=-1)  # Softmax to get probabilities for the current token
        token_prob = probs[0, token_id].item()  # Probability of the actual token

        # Update cumulative log probability for the full sequence up to this token
        log_prob_sum_full += np.log(token_prob + 1e-10)
        # Calculate and store cumulative confidence for this subsequence
        cumulative_confidences.append(np.exp(log_prob_sum_full))
        
        # Entropy calculation for the token
        entropy_sum -= token_prob * np.log(token_prob + 1e-10)

    # Full sequence confidence (cumulative up to the last token)
    P_T_given_I_Q_full = cumulative_confidences[-1] if cumulative_confidences else np.exp(log_prob_sum_full)

    # print(f"Overall confidence P(T | I, Q): {P_T_given_I_Q_full}")
    # print(f"Entropy H(T | I, Q): {entropy_sum}")
    # print("Per-token cumulative confidences:", cumulative_confidences)

    return P_T_given_I_Q_full, entropy_sum, cumulative_confidences




@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json        
        ## new args for recursive generation
        generation_type="default",
        fix_grid="default",
        attention_thresholding_type="layer_mean",
        attention_threshold="0.1",
        remove_unpadding=False,
        regenerate_condition="all",
        resized_image_size=168,
        positional_embedding_type="default",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # additional parameters for recursion
        self.generation_type = generation_type        
        self.fix_grid = fix_grid
        self.attention_thresholding_type = attention_thresholding_type
        self.attention_threshold = attention_threshold
        self.regenerate_condition = regenerate_condition

        print(f"generation_type: {generation_type}")
        print(f"fix_grid: {fix_grid}")
        print(f"attention_thresholding_type: {attention_thresholding_type}")
        print(f"attention_threshold: {attention_threshold}")
        print(f"regenerate_condition: {regenerate_condition}")
        
        ## default = "spatial_unpad" for llava1.6
        ## To remove unpadding, set remove_unpadding=True -> mm.path_merge_type will be 'spatial'
        if remove_unpadding == True:
            print("remove unpadding=True, change to 'spatial'")
            self._model.config.mm_patch_merge_type = "spatial"
            
        # CSV output path as a class attribute
        self.output_csv_path = Path("generation_output.csv")
        
        if not self.output_csv_path.exists():
            with open(self.output_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                
                # Basic headers
                headers = ["Doc ID", "Stage", "Text Output"]
                
                # Add headers for cumulative confidences dynamically
                headers += [f"Cumulative Confidence {i+1}" for i in range(10)]
                
                # Write header row
                writer.writerow(headers)

        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        
        self.resized_image_size = resized_image_size
        self.positional_embedding_type = positional_embedding_type
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.positional_embedding_type == "default":
            assert resized_image_size==336, "default embedding only allows size of 336"
        
        else:
            print(f"change positional embedding to {positional_embedding_type}")
            self.model.model.downsampled_vision_tower = copy.deepcopy(self.model.model.vision_tower)
            # Default configurations of model position embedding
            patch_size = 14
            num_patches = (resized_image_size // patch_size) ** 2
            num_positions = num_patches + 1
            embed_dim = self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.embed_dim

            self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.image_size = resized_image_size
            self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.num_patches = num_patches
            self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.num_positions = num_positions
            self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)

            # Modify positional embedding to match the resized image size
            if positional_embedding_type == "zero":       
                self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                torch.nn.init.constant_(self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight, 0)
            elif positional_embedding_type == "interpolation":
                # Interpolate from the pretrained positional embedding
                original_embedding = self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data
                original_num_positions = original_embedding.size(0)
                new_embedding = torch.nn.functional.interpolate(
                    original_embedding.unsqueeze(0).transpose(1, 2), 
                    size=(num_positions,), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
                self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                self.model.model.downsampled_ision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
            elif positional_embedding_type == "reduced":
                print("Reduced embedding type.")
                # Reduce the pretrained embedding by truncating
                original_embedding = self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data
                self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(original_embedding[:num_positions])
            self.model.to(device)
    
    # Method to log each stage's results
    def save_stage_to_csv(self, doc_id, stage, text_output, cumulative_confidences):
        with open(self.output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Prepare the row with doc_id, stage, and text_output, followed by each cumulative confidence as separate columns
            row = [doc_id, stage, text_output] + cumulative_confidences
            writer.writerow(row)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        cnt = 0
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]      

            ## original model accepts several diffrent grids (e.g. 1x2, 1x3, 2x2)
            ## for recursive implementation, we only use 2x2 grid (might be updated in future)
            ## Set grid to 2x2 for recursive generation, else default
            if self.fix_grid == "2x2":
                flattened_visuals = [make_square(visual, min_size=336) for visual in flattened_visuals]
            else:
                pass

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor, divide_shape = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    # interpolate for recursion
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    downsampled_image_tensor = F.interpolate(image_tensor[0], size=(image_tensor.shape[-2]//2, image_tensor.shape[-1]//2), mode='bilinear', align_corners=False)
                    # downsampled_image_tensor = F.interpolate(image_tensor[0], size=(self.resized_image_size, self.resized_image_size), mode='bilinear', align_corners=False)
                    downsampled_image_tensor = downsampled_image_tensor.unsqueeze(0)
                    downsampled_image_tensor = downsampled_image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            
            ## main generation part
            try:
            # if True:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    downsampled_image=downsampled_image_tensor,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    generation_type="downsampled",
                    return_dict_in_generate=True,
                    output_attentions=True,
                    # output_scores=True,
                )
                # text_outputs = self.tokenizer.batch_decode(cont["sequences"], skip_special_tokens=True) 
                text_outputs = [self.tokenizer.decode(cont["sequences"][0], skip_special_tokens=True).strip()]

                # text_outputs = self.tokenizer.decode(cont["sequences"][0], skip_special_tokens=True).strip()
                # print()
                # print(text_outputs)
                # print()
                # exit()
                    
                #################################################
                #################################################
                #################################################
                if "recursion" in self.generation_type:
                    #print("recursive 1st stage")
                    if cont["sequences"][0][0] == 1:
                        cont["sequences"] = cont["sequences"][0][1:].unsqueeze(0)
                    # print(cont['sequences'][0])
                    # scores = cont.scores                 

                    # # Calculate entropy and all cumulative confidences
                    # P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                    #     cont["sequences"][0], scores = scores
                    # )              
                    # # Save first stage results to CSV
                    # self.save_stage_to_csv("Stage 1", doc_id, text_outputs, cumulative_confidences)

                    # returns attention over image tokens
                    image = flattened_visuals[0]
                    # folder = f"/home/aidas_intern_1/woohyeon/lmms-woohyeon/vis"
                    # os.makedirs(folder, exist_ok=True)
                    # folder = f"/home/aidas_intern_1/woohyeon/lmms-woohyeon/vis/{str(cnt).zfill(6)}"
                    # cnt += 1
                    # os.makedirs(folder, exist_ok=True)

                    heat_torch_stack, ret_attn = get_heatmap(
                        self.model.get_downsampled_vision_tower(),
                        cont,
                        self.tokenizer,
                        question_input[0],
                        image,
                        input_ids,
                        # folder,
                    )
                    np_img = np.array(image)[:, :, ::-1]
                    
                    med = torch.stack(heat_torch_stack, dim=0)
                    med = med.mean(dim=0)
                    np_img = np.array(image)[:, :, ::-1]
                    for i, attn in enumerate(heat_torch_stack):
                        attn -= med
                        attn = torch.relu(attn)
                        attn = attn / attn.max()
                        img_with_attn, heatmap = show_mask_on_image(np_img, attn.numpy())
                        img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
                        # tt = self.tokenizer.decode(cont["sequences"][0][i]).strip()
                        # cv2.imwrite(f"{folder}/{str(i).zfill(3)}_{tt}.png", img_with_attn)

                    # Delete for memory management
                    del cont

                    ## averages attention over the layers to determine threshold
                    if self.attention_thresholding_type == "layer_mean":
                        med = torch.stack(ret_attn, dim=0)
                        med = med.mean(dim=0)
                        # [0] indicates the first token generated (change to [1] if output includes <s>)
                        attn = ret_attn[0] - med
                        attn = torch.relu(attn)
                        attn = attn / attn.max()

                        image_mask_list = []
                        for row in range(attn.shape[0]):
                            for col in range(attn.shape[1]):
                                if attn[row, col] > self.attention_threshold:
                                    image_mask_list.append(torch.LongTensor([[row, col]]))
                        if len(image_mask_list) == 0:
                            image_mask = None
                        else:
                            image_mask = torch.cat(image_mask_list)
                    else:
                        image_mask = None
                        
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        downsampled_image=downsampled_image_tensor,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        generation_type="recursion_stage1",
                        # return_dict_in_generate=True,
                        # output_attentions=True,
                        image_mask = image_mask,
                        # output_scores=True,
                    )
                    #print("recursive 2nd stage")
                    # print()
                    # print(text_outputs)
                    # print()
                #################################################
                #################################################
                #################################################
                    
                    
                    
                    # if cumulative_confidences[0] > 0.8:
                    #     print("score over threshold, do not recurse")
                    #     text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                    #     # Save second stage result as None to CSV
                    #     self.save_stage_to_csv("Stage 2", doc_id, ["No recurse"], cumulative_confidences)
                    # else:   
                    #     ## regenerate with image mask
                    #     ## remove output_attentions for efficient generation
                    #     cont = self.model.generate(
                    #         input_ids,
                    #         attention_mask=attention_masks,
                    #         pad_token_id=pad_token_ids,
                    #         images=image_tensor,
                    #         image_sizes=gen_kwargs["image_sizes"],
                    #         do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    #         temperature=gen_kwargs["temperature"],
                    #         top_p=gen_kwargs["top_p"],
                    #         num_beams=gen_kwargs["num_beams"],
                    #         max_new_tokens=gen_kwargs["max_new_tokens"],
                    #         use_cache=self.use_cache,
                    #         generation_type=self.generation_type,
                    #         return_dict_in_generate=True,
                    #         # output_attentions=True,
                    #         image_mask = image_mask,
                    #         output_scores=True,
                    #     )
                    #     #print("recursive 2nd stage")
                    #     text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                    #     #print(text_outputs)    
                        
                    #     P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                    #         cont["sequences"][0], scores = cont.scores
                    #     )
                        
                    #     # Save second stage results to CSV
                    #     self.save_stage_to_csv("Stage 2", doc_id, text_outputs, cumulative_confidences)

                ## no recursion: remove output_attentions, return_dict params since passing them requires additional memory
                # else:
                #     #print("no recursion")
                #     cont = self.model.generate(
                #         input_ids,
                #         attention_mask=attention_masks,
                #         pad_token_id=pad_token_ids,
                #         images=image_tensor,
                #         image_sizes=gen_kwargs["image_sizes"],
                #         do_sample=True if gen_kwargs["temperature"] > 0 else False,
                #         temperature=gen_kwargs["temperature"],
                #         top_p=gen_kwargs["top_p"],
                #         num_beams=gen_kwargs["num_beams"],
                #         max_new_tokens=gen_kwargs["max_new_tokens"],
                #         use_cache=self.use_cache,
                #         generation_type=self.generation_type,
                #         return_dict_in_generate=True,
                #         # output_attentions=True,
                #         output_scores=True
                #     )
                #     text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                #     #print(text_outputs)                     
                    
                #     # Calculate entropy and all cumulative confidences
                #     P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                #         cont["sequences"][0], cont.scores
                #     )
                #     # Save non-recursive results to CSV
                #     self.save_stage_to_csv("Non-recursive", doc_id, text_outputs, cumulative_confidences)



            except Exception as e:
                eval_logger.error(f"Error {e} in generating, generate with default")
                exit()
                cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        generation_type="default",
                        # return_dict_in_generate=True,
                        # output_attentions=True,
                    )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                #raise e
                # cont = ""
                # text_outputs = [""]

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res) 

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
