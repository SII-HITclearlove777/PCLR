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

import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from myllava.model.utils import auto_upgrade


def make_delta(base_model_path, target_model_path, delta_path, hub_repo_id):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    auto_upgrade(target_model_path)
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    for name, param in tqdm(target.state_dict().items()):
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name}'
            continue
        
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], f'{name}: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] -= bparam

    if hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": hub_repo_id}
    else:
        kwargs = {}

    target.save_pretrained(delta_path, **kwargs)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path, **kwargs)

