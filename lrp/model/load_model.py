import argparse
import torch
import os
import json

import transformers
from PIL import Image
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from myllava.model import *
from myllava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from lrp.model import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class ModelArguments_generate:
    vision_tower: Optional[str] = field(default=None)
    mm_vision_tower: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")



def load_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    llm_path = config["_name_or_path"]
    vit_path = config["mm_vision_tower"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_path, use_fast=False,)
    base_model = LlavaLlamaForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True, **kwargs)
    base_model.config.update(config)
    
    mm_use_im_start_end = config["mm_use_im_start_end"]
    mm_use_im_patch_token = config["mm_use_im_patch_token"]
    base_model.config.use_cache = False
    tokenizer.pad_token = tokenizer.unk_token
    pretrain_mm_mlp_adapter = os.path.join(model_path, 'mm_projector.bin')
    model_args = ModelArguments_generate(
        mm_vision_tower=vit_path,
        pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
        mm_vision_select_layer=config["mm_vision_select_layer"],
        mm_projector_type=config["mm_projector_type"],
        mm_use_im_start_end=mm_use_im_start_end,
        mm_use_im_patch_token=mm_use_im_patch_token,
        mm_patch_merge_type=config["mm_patch_merge_type"],
        mm_vision_select_feature=config["mm_vision_select_feature"]
    )
    base_model.get_model().initialize_vision_modules(model_args=model_args)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    base_model.resize_token_embeddings(len(tokenizer))
    vision_tower = base_model.get_vision_tower().to(device=base_model.device, dtype=base_model.dtype)
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    image_processor = vision_tower.image_processor
    
    
    if hasattr(base_model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    kwargs['mode'] = 'generate'
    model = LrpModel.from_pretrained(base_model, model_path, **kwargs)

    if model_path is not None:
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        missing_keys, unexpected_keys = model.load_state_dict(mm_projector_weights, strict=False)
        print("mm_projector unexpected keys: ", unexpected_keys)
        lrp_weights = torch.load(os.path.join(model_path, 'lrp.bin'), map_location='cpu')
        lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
        missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
        print("lrp unexpected keys: ", unexpected_keys)
    
    model.base_model.get_model().mm_projector.to(device=base_model.device, dtype=base_model.dtype)
    model.eval()
    return tokenizer, model, image_processor, context_len
