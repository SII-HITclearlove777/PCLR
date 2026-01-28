
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
from lrp_qwen.train.Lrp_trainer import LrpTrainer
from lrp_qwen.train.LrpTS_trainer import LrpTSTrainer

from QwenVL.Qwen_model.modeling_qwen import QWenLMHeadModel

from lrp_qwen.model import LrpModel, LrpTSModel, LrpConfig
from PIL import Image, ImageFile
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
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
    query_len: Optional[int] = field(default=256)
    
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
    # Loss weights
    loss_weight: float = field(
        default=None,
        metadata={"help": "Weight for the first loss component"}
    )
    # lrp 
    if_merge: Optional[bool] = field(default=True)
    Teacher_Student: Optional[bool] = field(default=False)
    skip_interval: Optional[int] = field(default=4)
    task_num: Optional[int] = field(default=None)
    # outside
    outside_qk: Optional[bool] = field(default=False)
    outside_llm_hidden: Optional[int] = field(default=None)
    outside_vit_hidden: Optional[int] = field(default=None)
        


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    prompt_lr: Optional[float] = None
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
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
        norm = torch.norm(new_vit_query_tensor, p=2)
        if norm < 0.01:
            new_vit_query = torch.zeros_like(new_vit_query_tensor)
        else:
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
                    to_return_ts[key] = merged_dict[key]
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
                    to_return_ts[key] = to_return[key]
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
    multimodal_keyword = 'visual'
    for name, module in model.named_modules():
        if multimodal_keyword in name:
            continue
            # if 'attn_pool' in name:
            #     pass
            # else:
            #     continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    target_list = list(lora_module_names)
    target_dic={"llm":[]}
    for target in target_list:
        target_dic["llm"].append(target)
    return target_dic


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    conv_roles = ["user", "assistant"]
    roles = {"human": "user", "gpt": "assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv_roles[0]:
            source = source[1:]

        input_id, target, instruction_ids = [], [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"].replace("<image>","<img></img>")).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == conv_roles[0]:
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == conv_roles[1]:
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        
        instruction = [source[0]["value"].replace("<image>\n",""),]
        instruction_id = tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        
        instruction_ids.append(instruction_id)
        input_ids.append(torch.tensor(input_id, dtype=torch.int))
        targets.append(torch.tensor(target, dtype=torch.int).long())
    return dict(
        input_ids=input_ids,
        instruction_ids=instruction_ids,
        labels=targets,
    )

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments, vit):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.vit = vit

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
            # image_folder = self.data_args.image_folder
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = Image.open(image_file).convert('RGB')
            image = self.vit.image_transform(image)
        sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             instruction_ids=data_dict["instruction_ids"][0],
                             labels=data_dict["labels"][0],
                             images=None,
                             has_image=None)
        if 'image' in self.list_data_dict[i].keys():
            data_dict['images'] = image
            data_dict['has_image'] = 1
        else:
            crop_size = self.vit.image_size
            data_dict['images'] = torch.zeros(3, crop_size[0], crop_size[1])
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
        images = [instance['images'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
        batch['has_image']=[instance['has_image'] for instance in instances]
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, vit) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args, vit=vit)
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

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    base_model = QWenLMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    
    base_model.transformer.requires_grad_(False)
    if hasattr(base_model,'transformer') and hasattr(base_model.transformer,'visual'):
        base_model.transformer.visual.requires_grad_(False)
        # if hasattr(base_model.transformer.visual,'attn_pool'):
        #     base_model.transformer.visual.attn_pool.requires_grad_(True)
        #     if hasattr(base_model.transformer.visual,'pos_embed'):
        #         base_model.transformer.visual.attn_pool.pos_embed.requires_grad_(False)
    img_token_span_size = model_args.query_len
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        img_token_span=img_token_span_size,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    llm_target_dic = find_all_linear_names(base_model)
    for single in llm_target_dic["llm"]:
        if 'visual' in single:
            print(single)
    if training_args.gradient_checkpointing:
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                input.requires_grad_(True)
            def make_ouputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_ouputs_require_grad)
            base_model.transformer.visual.ln_pre.register_forward_hook(make_inputs_require_grad)
            base_model.transformer.visual.attn_pool.ln_q.register_forward_hook(make_inputs_require_grad)

    lrp_config = LrpConfig(
        mode="train",
        device=training_args.device,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        llm_target_modules=llm_target_dic,
        llm_hidden_size=base_model.config.hidden_size,
        vit_hidden_size=base_model.config.visual["width"],
        llm_lora_alpha=lrp_args.llm_lora_alpha,
        llm_lora_dropout=lrp_args.llm_lora_dropout,
        llm_teacher_part1_rank_size=lrp_args.llm_teacher_part1_rank_size,
        llm_teacher_part2_rank_size=lrp_args.llm_teacher_part2_rank_size,
        llm_share_rank_size=lrp_args.llm_share_rank_size,
        llm_static_rank_size=lrp_args.llm_static_rank_size,
        llm_train_rank_size=lrp_args.llm_train_rank_size,
        llm_top_rank=lrp_args.llm_top_rank,
        llm_freeze_share=lrp_args.llm_freeze_share,
        task_num=lrp_args.task_num,
        loss_weight=lrp_args.loss_weight,
        outside_qk=lrp_args.outside_qk,
        outside_llm_hidden=lrp_args.outside_llm_hidden,
        outside_vit_hidden=lrp_args.outside_vit_hidden,
        use_loss_probability=lrp_args.use_loss_probability,
    )
    if lrp_args.Teacher_Student:
        model = LrpTSModel(base_model, lrp_config)
        if lrp_args.lrp_model_path is not None:
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
        
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, vit=base_model.transformer.visual)
        trainer = LrpTSTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
        trainer.train()
        trainer.save_state()
        model.config.use_cache = True
        output_dir = training_args.output_dir
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'train_keys_llm', 'train_keys_vit', 'train_prompt_pool', 'share_rank_A_pool', 'share_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save = get_lrp_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_merge = True, if_query = False)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.lrp_config.save_pretrained(output_dir, if_merge = True, task_num = lrp_args.task_num)
            torch.save(lrp_weight_to_save, os.path.join(output_dir, f'lrp.bin'))
    else:
        model = LrpModel(base_model, lrp_config)
        if lrp_args.lrp_model_path is not None:
            lrp_weights = torch.load(os.path.join(lrp_args.lrp_model_path, 'lrp.bin'), map_location='cpu')
            lrp_weights = {k: v.to(torch.float16) for k, v in lrp_weights.items()}
            missing_keys, unexpected_keys = model.load_state_dict(lrp_weights, strict=False)
            print("lrp unexpected keys: ", unexpected_keys)
        if lrp_args.load_key_init_path is not None:
            init_data = torch.load(lrp_args.load_key_init_path)
            model.init_key(init_data)
            if lrp_args.init_value:
                model.init_value()
        model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        for name, module in model.named_modules():
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16) 
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, vit=base_model.transformer.visual)
        trainer = LrpTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
        trainer.train()
        trainer.save_state()
        model.config.use_cache = True
        output_dir = training_args.output_dir
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'static_keys_llm', 'static_keys_vit', 'train_keys_llm', 'train_keys_vit', 'static_prompt_pool', 'train_prompt_pool', 
                                'share_rank_A_pool', 'share_rank_B_pool', 'static_rank_A_pool', 'static_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save_ts = get_lrpts_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_query = True)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.lrp_config.save_pretrained(output_dir, if_merge = lrp_args.if_merge, task_num = lrp_args.task_num + 1)
            torch.save(lrp_weight_to_save_ts, os.path.join(output_dir, f'lrp_ts.bin'))
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'static_keys_llm', 'static_keys_vit', 'train_keys_llm', 'train_keys_vit', 'static_prompt_pool', 'train_prompt_pool', 
                                'share_rank_A_pool', 'share_rank_B_pool', 'static_rank_A_pool', 'static_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save = get_lrp_state_maybe_zero_3(model.named_parameters(), lrp_keys_to_match, if_merge = lrp_args.if_merge, if_query = True)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            torch.save(lrp_weight_to_save, os.path.join(output_dir, f'lrp.bin'))


if __name__ == "__main__":
    train()
