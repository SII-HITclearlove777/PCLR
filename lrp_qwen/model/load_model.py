import argparse
import torch
import os
import json

import transformers
from PIL import Image
import math
from transformers import AutoTokenizer, AutoConfig
from QwenVL.Qwen_model.modeling_qwen import QWenLMHeadModel
from lrp_qwen.model import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers
from PIL import Image
import math
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from torch import nn


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_name):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name 
        self.select_layer = -2
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features
        return image_features

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class Outside_query_function(nn.Module):
    def __init__(self, llm_path, llm_embedding_path, vit_clip_path, device_map, use_bf16=True):
        super(Outside_query_function, self).__init__()
        config = transformers.AutoConfig.from_pretrained(llm_path)
        embedding_weights = torch.load(
            llm_embedding_path,
            map_location="cpu"
        )["model.embed_tokens.weight"]
        embedding_layer = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )
        embedding_layer.load_state_dict({"weight": embedding_weights})
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        vision_tower = CLIPVisionTower(vit_clip_path)
        vision_tower.load_model(device_map)
        self.vision_tower = vision_tower.to(dtype = (torch.bfloat16 if use_bf16 else torch.float32))
        self.image_processor = self.vision_tower.image_processor
        self.qdtype = self.vision_tower.dtype
        self.qdevice = self.vision_tower.device
        self.embedding_layer = embedding_layer.to(dtype = self.qdtype, device = self.qdevice)
        self.vision_tower.requires_grad_(False)
        self.vit_hidden_size = self.vision_tower.config.hidden_size
        
    
    def get_query(self, text, image_path):
        with torch.inference_mode():
            text_id = self.tokenizer(
                [text,],
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            text_embeding = self.embedding_layer(text_id[0].to(device=self.qdevice)).unsqueeze(0)
            llm_query = torch.mean(text_embeding, dim=1)
            llm_query = F.normalize(llm_query, dim=-1)
            if image_path != '':
                image = Image.open(image_path)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).to(dtype=self.qdtype, device=self.qdevice)
            else:
                images = None
            if images is not None:
                vit_query = self.vision_tower(images)[:,0]
                vit_query = F.normalize(vit_query, p=2, dim=-1)
            else:
                vit_query = torch.zeros(llm_query.shape[0], self.vit_hidden_size, dtype=self.qdtype, device=self.qdevice)
            return llm_query, vit_query
        

def load_model(model_path, device_map="auto", device="cuda", use_bf16=True, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config_ = json.load(config_file)
    base_path = config_["_name_or_path"]
    config = transformers.AutoConfig.from_pretrained(
        base_path,
        trust_remote_code=True,
    )
    base_model = QWenLMHeadModel.from_pretrained(
        base_path,
        config=config,
        torch_dtype=(torch.bfloat16 if use_bf16 else torch.float32),
        **kwargs,
    )

    kwargs['mode'] = 'generate'
    model = LrpModel.from_pretrained(base_model, model_path, **kwargs)
    if model_path is not None:
        lrp_weights = torch.load(os.path.join(model_path, 'lrp.bin'), map_location='cpu')
        lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
        missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
        print("lrp unexpected keys: ", unexpected_keys)
    model.eval()
    img_token_span_size = 256
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_path, img_token_span=img_token_span_size, use_fast=False, trust_remote_code=True)
    
    query_cls = Outside_query_function(llm_path = "./vicuna-7b-v1.5", 
                                   llm_embedding_path = "./vicuna-7b-v1.5/pytorch_model-00001-of-00002.bin", 
                                   vit_clip_path = "./clip-vit-large-patch14-336", 
                                   device_map = device_map, use_bf16 = use_bf16)
    
    return model, tokenizer, query_cls
