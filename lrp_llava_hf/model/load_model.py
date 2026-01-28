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

from lrp_llava_hf.model import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List



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

    base_model = transformers.LlavaForConditionalGeneration.from_pretrained(llm_path, low_cpu_mem_usage=True).to("cuda")
    image_processor = transformers.AutoImageProcessor.from_pretrained(llm_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    base_model.config.update(config)
    base_model.config.use_cache = False
    tokenizer.pad_token = tokenizer.unk_token

    kwargs['mode'] = 'generate'
    model = LrpModel.from_pretrained(base_model, model_path, **kwargs)
    if model_path is not None:
        lrp_weights = torch.load(os.path.join(model_path, 'lrp.bin'), map_location='cpu')
        lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
        missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
        print("lrp unexpected keys: ", unexpected_keys)

    model.eval()
    return tokenizer, model, image_processor
