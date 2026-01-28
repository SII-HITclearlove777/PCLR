#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from myllava.model import *
from myllava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from huggingface_hub import hf_hub_download
from peft import PeftModel 


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
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

    if 'llava' in model_name.lower():
        if 'lora' in model_name.lower() and model_base is None:
            # 如果模型名包含 lora，但没有提供基础模型路径，发出警告
            warnings.warn('检测到模型名称中有 `lora`，但未提供 `model_base`。如果加载的是 LoRA 模型，请提供 `model_base` 参数。')
        elif 'lora' in model_name.lower() and model_base is not None:
            # 加载带有 LoRA 权重的 LLaVA 模型
            from llava.model.language_model.llava_llama import LlavaConfig
            # 加载 LLaVA 的配置
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            # 加载分词器，禁用快速分词器以避免潜在问题
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

            print('从基础模型加载 LLaVA...')
            # 加载 LLaVA 模型的基础部分
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            print('加载额外的 LLaVA 权重...')
            # 尝试加载本地或 Hugging Face Hub 上的非 LoRA 可训练权重
            non_lora_trainables_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_trainables_path):
                non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
            else:
                non_lora_trainables = torch.load(hf_hub_download(repo_id=model_path, filename='non_lora_trainables.bin'), map_location='cpu')

            # 调整权重键名以匹配模型结构
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            # 加载非 LoRA 可训练权重
            model.load_state_dict(non_lora_trainables, strict=False)

            print('加载 LoRA 权重...')
            # 加载并合并 LoRA 权重
            model = PeftModel.from_pretrained(model, model_path)
            print('合并 LoRA 权重...')
            model = model.merge_and_unload()
            print('模型已加载...')
        elif model_base is not None:
            # 可能仅包含多模态投影层
            print('从基础模型加载 LLaVA...')
            if 'mpt' in model_name.lower():
                # 如果是 MPT 模型，确保有正确的配置文件，并加载相应的分词器和配置
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                # 对于其他类型的 LLaVA 模型，加载分词器和配置
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            print('加载多模态投影层权重...')
            # 加载多模态投影层权重，并转换为半精度浮点数
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # 直接加载 LLaVA 模型，处理不同变体
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # 加载纯语言模型，处理 PEFT 模型和普通模型两种情况
        if model_base is not None:
            # PEFT 模型，加载基础模型、LoRA 权重，并合并
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"从 {model_path} 加载 LoRA 权重")
            model = PeftModel.from_pretrained(model, model_path)
            print("合并权重")
            model = model.merge_and_unload()
            print('转换为 FP16...')
            model.to(torch.float16)
        else:
            # 普通模型，直接加载模型和分词器
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None




    if 'llava' in model_name.lower():
        # 配置多模态支持，添加特定的图像标记符
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        # 根据新的分词器调整模型的嵌入层大小
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            # 加载视觉塔模型，如果指定了具体的设备映射则移动到该设备
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    # 确定上下文长度，优先使用模型配置中的值
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len







