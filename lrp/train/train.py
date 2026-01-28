
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from myllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from lrp.train.Lrp_trainer import LrpTrainer
from lrp.train.LrpTS_trainer import LrpTSTrainer
from myllava import conversation as conversation_lib
from myllava.model import *
from myllava.mm_utils import tokenizer_image_token
from lrp.model import LrpModel, LrpTSModel, LrpConfig
from PIL import Image, ImageFile
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
import random
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class LrpArguments:
    lrp_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained lrp model"}
    )
    load_key_init_path: Optional[str] = field(
        default=None,
    )
    use_loss_probability: Optional[bool] = field(default=False)
    init_value: Optional[bool] = field(default=False)
    init_near: Optional[bool] = field(default=False)
    vit_prompt_tune: Optional[bool] = field(default=False)
    mm_zip_tune: Optional[bool] = field(default=True)
    # LLM LoRA parameters
    llm_lora_alpha: Optional[int] = field(
        default=256,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    llm_lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    # LLM Rank sizes
    llm_share_rank_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of shared rank"}
    )
    llm_teacher_part1_rank_size: Optional[int] = field(default=None)
    llm_teacher_part2_rank_size: Optional[int] = field(default=None)
    llm_static_rank_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of static rank"}
    )
    llm_train_rank_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of trainable rank"}
    )
    llm_top_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k rank selection"}
    )
    llm_freeze_share: Optional[bool] = field(default=False)
    # VIT prompt sizes
    vit_prompt_alpha: Optional[int] = field(default=64)
    vit_teacher_part1_prompt_size: Optional[int] = field(default=None)
    vit_teacher_part2_prompt_size: Optional[int] = field(default=None)
    vit_static_prompt_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of static prompts for vision transformer"}
    )
    vit_train_prompt_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of trainable prompts for vision transformer"}
    )
    vit_prompt_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of prompts for vision transformer"}
    )
    vit_top_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k rank selection"}
    )
    # mm LoRA parameters
    mm_static_rank_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of static rank"}
    )
    mm_train_rank_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of trainable rank"}
    )
    # Loss weights
    loss_weight1: float = field(
        default=None,
        metadata={"help": "Weight for the first loss component"}
    )
    loss_weight2: float = field(
        default=None,
        metadata={"help": "Weight for the second loss component"}
    )
    loss_weight3: float = field(
        default=None,
        metadata={"help": "Weight for the third loss component"}
    )
    loss_weight4: float = field(
        default=None,
        metadata={"help": "Weight for the 4th loss component"}
    )
    # lrp 
    if_merge: Optional[bool] = field(default=True)
    Teacher_Student: Optional[bool] = field(default=False)
    skip_interval: Optional[int] = field(default=4)
    task_num: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    mm_projector_lr: Optional[float] = None
    prompt_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        if hasattr(param, 'ds_active_sub_modules') and param.ds_active_sub_modules:
            param.ds_active_sub_modules.clear()
        with zero.GatheredParameters([param], enabled=True):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    grad = {k: t.requires_grad for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    new_dic = {}
    for key in to_return.keys():
        if 'weight' in key or 'bias' in key:
            new_dic[key] = to_return[key]
    return new_dic


def get_lrpts_state_maybe_zero_3(named_params, keys_to_match, if_query = True, mm_target_dic = None):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    grad = {k: t.requires_grad for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    if if_query:
        llm_query_list = []
        vit_query_list = []
        for key, value in to_return.items():
            if "train_keys_llm" in key:
                llm_query_list.append(to_return[key])
            if "train_keys_vit" in key:
                vit_query_list.append(to_return[key])
        new_llm_query_tensor = torch.mean(torch.cat(llm_query_list, dim=0), dim=0).unsqueeze(0)
        new_llm_query = torch.nn.functional.normalize(new_llm_query_tensor, p=2, dim=1)
        new_vit_query_tensor = torch.mean(torch.cat(vit_query_list, dim=0), dim=0).unsqueeze(0)
        new_vit_query = torch.nn.functional.normalize(new_vit_query_tensor, p=2, dim=1)
        if "task_llm_query_pool" not in to_return.keys() or to_return["task_llm_query_pool"] is None:
            to_return["task_llm_query_pool"] = new_llm_query
        else:
            to_return["task_llm_query_pool"] = torch.cat([to_return["task_llm_query_pool"], new_llm_query], dim=0)
        if "task_vit_query_pool" not in to_return.keys() or to_return["task_vit_query_pool"] is None:
            to_return["task_vit_query_pool"] = new_vit_query
        else:
            to_return["task_vit_query_pool"] = torch.cat([to_return["task_vit_query_pool"], new_vit_query], dim=0)
    to_return_ts = {}
    if mm_target_dic is not None:
        no_teach_moudle = []
        no_teach_key = []
        for key in mm_target_dic.keys():
            no_teach_key.append(key)
            for s in mm_target_dic[key]:
                no_teach_moudle.append(s)
        merge_pairs = []
        processed_keys = set()
        for key in to_return.keys():
            if key in processed_keys:
                continue
            ls = key.split('.')
            if len(ls) == 2:
                layer = ls[-1]
                if 'static_' in key and any(layer == m for m in no_teach_key):
                    corresponding_train_key = key.replace('static_', 'train_')
                    merge_pairs.append((key, corresponding_train_key))
                    processed_keys.add(key)
                    processed_keys.add(corresponding_train_key)
                elif 'train_' in key and any(layer == m for m in no_teach_key):
                    corresponding_static_key = key.replace('train_', 'static_')
                    merge_pairs.append((corresponding_static_key, key))
                    processed_keys.add(key)
                    processed_keys.add(corresponding_static_key)
            else:
                if 'static_' in key and any(m in key for m in no_teach_moudle):
                    corresponding_train_key = key.replace('static_', 'train_')
                    merge_pairs.append((key, corresponding_train_key))
                    processed_keys.add(key)
                    processed_keys.add(corresponding_train_key)
                elif 'train_' in key and any(m in key for m in no_teach_moudle):
                    corresponding_static_key = key.replace('train_', 'static_')
                    merge_pairs.append((corresponding_static_key, key))
                    processed_keys.add(key)
                    processed_keys.add(corresponding_static_key)
        merged_dict = {}
        for static_key, train_key in merge_pairs:
            static_tensor = to_return.get(static_key)
            train_tensor = to_return.get(train_key)
            if static_tensor is not None and train_tensor is not None:
                if 'rank_A' in static_key:
                    merged_tensor = torch.cat([static_tensor, train_tensor], dim=1)
                else:
                    merged_tensor = torch.cat([static_tensor, train_tensor], dim=0)
                merged_dict[static_key] = merged_tensor
            elif static_tensor is not None:
                merged_dict[static_key] = static_tensor
            elif train_tensor is not None:
                merged_dict[static_key] = train_tensor

        for key in to_return.keys():
            ls = key.split('.')
            if len(ls) == 2:
                layer = ls[-1]
                if 'static_' in key and not any(layer == m for m in no_teach_key):
                    new_key = key.replace('static_', 'teacher_part1_')
                    to_return_ts[new_key] = to_return[key]
                elif 'train_' in key and not any(layer == m for m in no_teach_key):
                    new_key = key.replace('train_', 'teacher_part2_')
                    to_return_ts[new_key] = to_return[key]
                elif 'share_' in key and not any(layer == m for m in no_teach_key):
                    new_key = key.replace('share_', 'teacher_share_')
                    to_return_ts[new_key] = to_return[key]
                elif any(layer == m for m in no_teach_key):
                    if "train" in key:
                        to_return_ts[key.replace("train", "static")] = merged_dict[key.replace("train", "static")]
                    else:
                        continue
                else:
                    to_return_ts[key] = to_return[key]
            else:
                if 'static_' in key and not any(m in key for m in no_teach_moudle):
                    new_key = key.replace('static_', 'teacher_part1_')
                    to_return_ts[new_key] = to_return[key]
                elif 'train_' in key and not any(m in key for m in no_teach_moudle):
                    new_key = key.replace('train_', 'teacher_part2_')
                    to_return_ts[new_key] = to_return[key]
                elif 'share_' in key and not any(m in key for m in no_teach_moudle):
                    new_key = key.replace('share_', 'teacher_share_')
                    to_return_ts[new_key] = to_return[key]
                elif any(m in key for m in no_teach_moudle):
                    if "train" in key:
                        to_return_ts[key.replace("train", "static")] = merged_dict[key.replace("train", "static")]
                    else:
                        continue
                else:
                    to_return_ts[key] = to_return[key]
    else:
        for key in to_return.keys():
            if 'static_' in key:
                new_key = key.replace('static_', 'teacher_part1_')
                to_return_ts[new_key] = to_return[key]
            elif 'train_' in key:
                new_key = key.replace('train_', 'teacher_part2_')
                to_return_ts[new_key] = to_return[key]
            elif 'share_' in key:
                new_key = key.replace('share_', 'teacher_share_')
                to_return_ts[new_key] = to_return[key]
            else:
                to_return_ts[key] = to_return[key]
    return to_return_ts


def get_lrp_state_maybe_zero_3(named_params, keys_to_match, if_merge = False, if_query = True):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    grad = {k: t.requires_grad for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    if if_query:
        llm_query_list = []
        vit_query_list = []
        for key, value in to_return.items():
            if "train_keys_llm" in key:
                llm_query_list.append(to_return[key])
            if "train_keys_vit" in key:
                vit_query_list.append(to_return[key])
        new_llm_query_tensor = torch.mean(torch.cat(llm_query_list, dim=0), dim=0).unsqueeze(0)
        new_llm_query = torch.nn.functional.normalize(new_llm_query_tensor, p=2, dim=1)
        new_vit_query_tensor = torch.mean(torch.cat(vit_query_list, dim=0), dim=0).unsqueeze(0)
        new_vit_query = torch.nn.functional.normalize(new_vit_query_tensor, p=2, dim=1)
        if "task_llm_query_pool" not in to_return.keys() or to_return["task_llm_query_pool"] is None:
            to_return["task_llm_query_pool"] = new_llm_query
        else:
            to_return["task_llm_query_pool"] = torch.cat([to_return["task_llm_query_pool"], new_llm_query], dim=0)
        if "task_vit_query_pool" not in to_return.keys() or to_return["task_vit_query_pool"] is None:
            to_return["task_vit_query_pool"] = new_vit_query
        else:
            to_return["task_vit_query_pool"] = torch.cat([to_return["task_vit_query_pool"], new_vit_query], dim=0)
    if not if_merge:
        return to_return
    merge_pairs = []
    processed_keys = set()
    for key in to_return.keys():
        if key in processed_keys:
            continue
        if 'static_' in key:
            corresponding_train_key = key.replace('static_', 'train_')
            merge_pairs.append((key, corresponding_train_key))
            processed_keys.add(key)
            processed_keys.add(corresponding_train_key)
        elif 'train_' in key:
            corresponding_static_key = key.replace('train_', 'static_')
            merge_pairs.append((corresponding_static_key, key))
            processed_keys.add(key)
            processed_keys.add(corresponding_static_key)
    merged_dict = {}
    for static_key, train_key in merge_pairs:
        static_tensor = to_return.get(static_key)
        train_tensor = to_return.get(train_key)
        if static_tensor is not None and train_tensor is not None:
            if 'rank_A' in static_key:
                merged_tensor = torch.cat([static_tensor, train_tensor], dim=1)
            else:
                merged_tensor = torch.cat([static_tensor, train_tensor], dim=0)
            merged_dict[static_key] = merged_tensor
        elif static_tensor is not None:
            merged_dict[static_key] = static_tensor
        elif train_tensor is not None:
            merged_dict[static_key] = train_tensor
    unmerged_keys = set(to_return.keys()) - processed_keys
    for key in unmerged_keys:
        merged_dict[key] = to_return[key]
    save_dict = {}
    for key in merged_dict.keys():
        if "teacher_" not in key:
            save_dict[key] = merged_dict[key]
    return save_dict

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    target_list = list(lora_module_names)
    target_dic={}
    target_dic['mm_projector']=[]
    for i in range(model.config.num_hidden_layers):
        target_dic[str(i)]=[]
    for target in target_list:
        if 'mm_projector' not in target:
            target_dic[target.split('.')[2]].append(target)
        else:
            target_dic['mm_projector'].append(target)
    return target_dic


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    instructions = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        # instruction = source[0]["instruction"]
        instruction = source[0]["value"].replace("<image>\n","")
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        instructions.append(instruction)
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    instruction_ids = tokenizer(
            instructions,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        instruction_ids=instruction_ids,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    return preprocess_v1(sources, tokenizer, has_image=has_image)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
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
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0],
                            instruction_ids=data_dict["instruction_ids"][0])
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['has_image'] = 1
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['has_image'] = 0
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, instruction_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "instruction_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        instruction_ids = torch.nn.utils.rnn.pad_sequence(
            instruction_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            instruction_ids=instruction_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            instruction_mask=instruction_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['has_image']=[instance['has_image'] for instance in instances]
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, LrpArguments, DataArguments, TrainingArguments))

    model_args, lrp_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    base_model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )

    base_model.config.use_cache = False
    if model_args.freeze_backbone:
        base_model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        base_model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    if training_args.gradient_checkpointing:
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        
    if model_args.vision_tower is not None:
        base_model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = base_model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        base_model.config.image_aspect_ratio = data_args.image_aspect_ratio
        base_model.config.tokenizer_padding_side = tokenizer.padding_side
        base_model.config.tokenizer_model_max_length = tokenizer.model_max_length
        base_model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            base_model.requires_grad_(False)
            for p in base_model.get_model().mm_projector.parameters():
                p.requires_grad = True
        base_model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in base_model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.bits in [4, 8]:
            base_model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        base_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        base_model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        base_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        base_model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    llm_target_dic = find_all_linear_names(base_model)
    
    
    mm_target_dic = {}
    if lrp_args.vit_prompt_tune:
        vit_target_dic = {"vit":[ "model.vision_tower.vision_tower.vision_model.pre_layrnorm"]}
        def make_inputs_require_grad(module, input, output):
            input.requires_grad_(True)
        base_model.model.vision_tower.vision_tower.vision_model.pre_layrnorm.register_forward_hook(make_inputs_require_grad)
    else:
        vit_target_dic = {}
        
    if not lrp_args.mm_zip_tune:
        new_llm_target_dic = {}
        # l = ['0', '1', '2', '3', '4', '5', '6', '7']
        # l = ['8', '9', '10', '11', '12', '13', '14', '15']
        # l = ['16', '17', '18', '19', '20', '21', '22', '23']
        # l = ['24', '25', '26', '27', '28', '29', '30', '31']
        l = ['32', '33', '34', '35', '36', '37', '38', '39']
        for key in llm_target_dic.keys():
            if key in l:
                mm_target_dic[key] = llm_target_dic[key]
            else:
                new_llm_target_dic[key] = llm_target_dic[key]
        llm_target_dic = new_llm_target_dic
    else:
        mm_target_dic={}
    
    lrp_config = LrpConfig(
        mode="train",
        device=training_args.device,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        llm_target_modules=llm_target_dic,
        llm_hidden_size=base_model.config.hidden_size,
        vit_target_modules=vit_target_dic,
        vit_hidden_size=base_model.get_model().vision_tower.vision_tower.config.hidden_size,
        mm_target_modules=mm_target_dic,
        llm_lora_alpha=lrp_args.llm_lora_alpha,
        llm_lora_dropout=lrp_args.llm_lora_dropout,
        llm_teacher_part1_rank_size=lrp_args.llm_teacher_part1_rank_size,
        llm_teacher_part2_rank_size=lrp_args.llm_teacher_part2_rank_size,
        llm_share_rank_size=lrp_args.llm_share_rank_size,
        llm_static_rank_size=lrp_args.llm_static_rank_size,
        llm_train_rank_size=lrp_args.llm_train_rank_size,
        llm_top_rank=lrp_args.llm_top_rank,
        llm_freeze_share=lrp_args.llm_freeze_share,
        vit_prompt_alpha=lrp_args.vit_prompt_alpha,
        vit_teacher_part1_prompt_size=lrp_args.vit_teacher_part1_prompt_size,
        vit_teacher_part2_prompt_size=lrp_args.vit_teacher_part2_prompt_size,
        vit_static_prompt_size=lrp_args.vit_static_prompt_size,
        vit_train_prompt_size=lrp_args.vit_train_prompt_size,
        vit_prompt_len=lrp_args.vit_prompt_len,
        vit_top_rank=lrp_args.vit_top_rank,
        mm_lora_alpha=lrp_args.llm_lora_alpha,
        mm_lora_dropout=lrp_args.llm_lora_dropout,
        mm_static_rank_size=lrp_args.mm_static_rank_size,
        mm_train_rank_size=lrp_args.mm_train_rank_size,
        mm_top_rank=lrp_args.llm_top_rank,
        task_num=lrp_args.task_num,
        loss_weight1=lrp_args.loss_weight1,
        loss_weight2=lrp_args.loss_weight2,
        loss_weight3=lrp_args.loss_weight3,
        loss_weight4=lrp_args.loss_weight4,
        use_loss_probability=lrp_args.use_loss_probability,
    )
    if lrp_args.Teacher_Student:
        model = LrpTSModel(base_model, lrp_config)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter name: {name}, Parameter shape: {param.shape}, grad: {param.requires_grad}, device: {param.device}, dtype: {param.dtype}")
        if lrp_args.lrp_model_path is not None:
            mm_projector_weights = torch.load(os.path.join(lrp_args.lrp_model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            missing_keys, unexpected_keys = model.load_state_dict(mm_projector_weights, strict=False)
            print("mm_projector unexpected keys: ", unexpected_keys)
            lrp_weights = torch.load(os.path.join(lrp_args.lrp_model_path, 'lrp_ts.bin'), map_location='cpu')
            lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
            missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
            print("lrp unexpected keys: ", unexpected_keys)
        model.init_key_value(lrp_args.skip_interval)
        model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
            if 'query_pool' in name or 'query_pool' in name or 'keys' in name:
                module = module.to(torch.float32)
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)
        trainer = LrpTSTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
        trainer.train()
        trainer.save_state()
        model.config.use_cache = True
        output_dir = training_args.output_dir
        mm_keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(training_args, "use_im_start_end", False):
            mm_keys_to_match.extend(['embed_tokens', 'embed_in'])
        mm_weight_to_save = get_mm_adapter_state_maybe_zero_3(model.named_parameters(), mm_keys_to_match)
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'train_keys_llm', 'train_keys_vit', 'train_prompt_pool', 'share_rank_A_pool', 'share_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool',
                             'static_keys_llm', 'static_keys_vit', 'static_rank_A_pool', 'static_rank_B_pool']
        lrp_weight_to_save = get_lrp_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_merge = True, if_query = False)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.lrp_config.save_pretrained(output_dir, if_merge = True, task_num = lrp_args.task_num)
            torch.save(mm_weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            torch.save(lrp_weight_to_save, os.path.join(output_dir, f'lrp.bin'))
    else:
        model = LrpModel(base_model, lrp_config)
        if lrp_args.lrp_model_path is not None:
            mm_projector_weights = torch.load(os.path.join(lrp_args.lrp_model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            missing_keys, unexpected_keys = model.load_state_dict(mm_projector_weights, strict=False)
            print("mm_projector unexpected keys: ", unexpected_keys)
            lrp_weights = torch.load(os.path.join(lrp_args.lrp_model_path, 'lrp.bin'), map_location='cpu')
            lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
            missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
            print("lrp unexpected keys: ", unexpected_keys)
        if lrp_args.load_key_init_path is not None:
            init_data = torch.load(lrp_args.load_key_init_path)
            model.init_key(init_data)
            if lrp_args.init_value:
                model.init_value(lrp_args.init_near)
        model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16) 
                
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                data_args=data_args)
        trainer = LrpTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
        trainer.train()
        trainer.save_state()
        model.config.use_cache = True
        output_dir = training_args.output_dir
        mm_keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(training_args, "use_im_start_end", False):
            mm_keys_to_match.extend(['embed_tokens', 'embed_in'])
        mm_weight_to_save = get_mm_adapter_state_maybe_zero_3(model.named_parameters(), mm_keys_to_match)
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'static_keys_llm', 'static_keys_vit', 'train_keys_llm', 'train_keys_vit', 'static_prompt_pool', 'train_prompt_pool', 
                                'share_rank_A_pool', 'share_rank_B_pool', 'static_rank_A_pool', 'static_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save_ts = get_lrpts_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_query = True, mm_target_dic=mm_target_dic)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.lrp_config.save_pretrained(output_dir, if_merge = lrp_args.if_merge, task_num = lrp_args.task_num + 1)
            torch.save(mm_weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            torch.save(lrp_weight_to_save_ts, os.path.join(output_dir, f'lrp_ts.bin'))
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'static_keys_llm', 'static_keys_vit', 'train_keys_llm', 'train_keys_vit', 'static_prompt_pool', 'train_prompt_pool', 
                                'share_rank_A_pool', 'share_rank_B_pool', 'static_rank_A_pool', 'static_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save = get_lrp_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_merge = lrp_args.if_merge, if_query = True)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            torch.save(lrp_weight_to_save, os.path.join(output_dir, f'lrp.bin'))


if __name__ == "__main__":
    train()
    
    
    
    
