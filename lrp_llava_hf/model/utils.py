import math
from dataclasses import dataclass, field
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import os
import json
from torch.cuda.amp import autocast
import math

def get_llm_query(input_embedding, input_mask):
    output = input_embedding
    if input_mask is not None:
        masked_embdings = output * input_mask.unsqueeze(-1)
        valid_sums = masked_embdings.sum(dim=1)
        valid_lengths = input_mask.sum(dim=1).unsqueeze(-1)
        valid_lengths = valid_lengths.clamp(min=1)
        query = valid_sums / valid_lengths
        query = F.normalize(query, p=2, dim=-1)
    else:
        query = torch.mean(output, dim=1)
        query = F.normalize(query, p=2, dim=-1)
    return query



def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
            else:
                for param in target.parameters():
                    param.requires_grad = True
                setattr(parent, target_name, ModulesToSaveWrapper(target, adapter_name))

@dataclass
class LrpConfig:
    mode: str = field(default="train")  # "train" or "generate"
    device: torch.device = field(default=torch.device("cpu"))
    dtype: torch.dtype = field(default=torch.float32)
    task_num: Optional[int] = field(default=None)
    use_loss_probability: Optional[bool] = field(default=False)
    # LLM
    llm_target_modules: Optional[Dict] = field(default=None)
    llm_lora_alpha: Optional[int] = field(default=256)
    llm_lora_dropout: float = field(default=0.0)
    llm_share_rank_size: Optional[int] = field(default=None)
    llm_teacher_part1_rank_size: Optional[int] = field(default=None)
    llm_teacher_part2_rank_size: Optional[int] = field(default=None)
    llm_static_rank_size: Optional[int] = field(default=None)
    llm_train_rank_size: Optional[int] = field(default=None)
    llm_hidden_size: Optional[int] = field(default=None)
    llm_top_rank: Optional[int] = field(default=None)
    llm_freeze_share: Optional[bool] = field(default=False)
    # VIT
    vit_hidden_size: Optional[int] = field(default=None)
    # mm
    mm_target_modules: Optional[Dict] = field(default=None)
    mm_lora_alpha: Optional[int] = field(default=32)
    mm_lora_dropout: float = field(default=0.0)
    mm_share_rank_size: Optional[int] = field(default=None)
    mm_static_rank_size: Optional[int] = field(default=None)
    mm_train_rank_size: Optional[int] = field(default=None)
    mm_top_rank: Optional[int] = field(default=None)
    # loss
    loss_weight1: float = field(default=None)

    
    def __post_init__(self):
        self._validate_config()


    def _validate_config(self):
        if self.mode not in ["train", "generate"]:
            raise ValueError("mode must be either 'train' or 'generate'")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device instance")

    def to_dict(self) -> Dict:
        config_dict = {
            "mode": self.mode,
            "device": str(self.device),
            "dtype": str(self.dtype).split(".")[-1],

            "llm_target_modules" :self.llm_target_modules,
            "llm_lora_alpha" :self.llm_lora_alpha,
            "llm_lora_dropout" :self.llm_lora_dropout,
            "llm_share_rank_size" :self.llm_share_rank_size,
            "llm_static_rank_size" :self.llm_static_rank_size,
            "llm_train_rank_size" :self.llm_train_rank_size,
            "llm_hidden_size" :self.llm_hidden_size,
            "llm_top_rank" :self.llm_top_rank,
            
            "vit_hidden_size" :self.vit_hidden_size,
            
            "mm_target_modules" :self.mm_target_modules,
            "mm_lora_alpha" :self.mm_lora_alpha,
            "mm_lora_dropout" :self.mm_lora_dropout,
            "mm_share_rank_size" :self.mm_share_rank_size,
            "mm_static_rank_size" :self.mm_static_rank_size,
            "mm_train_rank_size" :self.mm_train_rank_size,
            "mm_top_rank" :self.mm_top_rank,
        }

        return config_dict

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config_name) -> "LrpConfig":
        if not pretrained_model_name_or_path:
            raise ValueError("pretrained_model_name_or_path cannot be empty")
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Path does not exist: {pretrained_model_name_or_path}")
        config_path = os.path.join(pretrained_model_name_or_path, config_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Can't find config file: {config_name} in {pretrained_model_name_or_path}")
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing config file {config_path}: {str(e)}")

        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = torch.device(config_dict["device"])
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            dtype_str = config_dict["dtype"]
            dtype_mapping = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            if dtype_str in dtype_mapping:
                config_dict["dtype"] = dtype_mapping[dtype_str]
            else:
                try:
                    config_dict["dtype"] = getattr(torch, dtype_str)
                except AttributeError:
                    raise ValueError(f"Invalid dtype: {dtype_str}")
        config = cls(**config_dict)
        return config


    def save_pretrained(self, save_directory: str, if_merge = False, task_num = 0):
        os.makedirs(save_directory, exist_ok=True)
        config_dict = self.to_dict()
        with open(os.path.join(save_directory, "lrp_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        if self.mode == "train":
            config_dict["mode"] = "generate"
            if if_merge:
                if config_dict["llm_static_rank_size"] is not None and config_dict["llm_train_rank_size"] is not None:
                    config_dict["llm_static_rank_size"] = config_dict["llm_static_rank_size"] + config_dict["llm_train_rank_size"]
                elif config_dict["llm_static_rank_size"] is not None:
                    config_dict["llm_static_rank_size"] = config_dict["llm_static_rank_size"]
                else:
                    config_dict["llm_static_rank_size"] = config_dict["llm_train_rank_size"]

                if config_dict["mm_static_rank_size"] is not None and config_dict["mm_train_rank_size"] is not None:
                    config_dict["mm_static_rank_size"] = config_dict["mm_static_rank_size"] + config_dict["mm_train_rank_size"]
                elif config_dict["mm_static_rank_size"] is not None:
                    config_dict["mm_static_rank_size"] = config_dict["mm_static_rank_size"]
                else:
                    config_dict["mm_static_rank_size"] = config_dict["mm_train_rank_size"]
                config_dict.pop("llm_train_rank_size")
                config_dict.pop("mm_train_rank_size")
            if task_num > 0:
                config_dict["task_num"] = task_num
            with open(os.path.join(save_directory, "lrp_generate_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "LrpConfig":
        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = torch.device(config_dict["device"])
        return cls(**config_dict)

    def merge_from(self, other: "LrpConfig") -> "LrpConfig":
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        for key, value in other_dict.items():
            if value is not None:
                merged_dict[key] = value
        return self.from_dict(merged_dict)


class LlmLrpTSLinear(nn.Linear):
    def __init__(
        self,
        pool_name,
        in_features: int,
        out_features: int,
        share_size, 
        teacher_part1_size,
        teacher_part2_size,
        train_size,
        top_rank,
        freeze_share,
        ori_device, 
        ori_dtype,
        get_score,
        get_state,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.pool_name = pool_name
        self.get_score = get_score
        self.get_state = get_state
        self.lora_alpha = lora_alpha
        self.teacher_part1_size = teacher_part1_size
        self.teacher_part2_size = teacher_part2_size
        self.share_size = share_size
        self.train_size = train_size
        self.top_rank = top_rank
        self.weight.requires_grad = False
        nn.Linear.reset_parameters(self)
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = None
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        if share_size is not None and share_size>0:
            teacher_share_rank_A_pool_tensor = torch.empty(in_features, share_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(teacher_share_rank_A_pool_tensor, mean=0.0, std=bound)
            teacher_share_rank_B_pool_tensor = torch.zeros(share_size, out_features, device=ori_device, dtype=ori_dtype)
            self.teacher_share_rank_A_pool = nn.Parameter(teacher_share_rank_A_pool_tensor, requires_grad=False)
            self.teacher_share_rank_B_pool = nn.Parameter(teacher_share_rank_B_pool_tensor, requires_grad=False)
            share_rank_A_pool_tensor = torch.empty(in_features, share_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(share_rank_A_pool_tensor, mean=0.0, std=bound)
            share_rank_B_pool_tensor = torch.zeros(share_size, out_features, device=ori_device, dtype=ori_dtype)
            self.share_rank_A_pool = nn.Parameter(share_rank_A_pool_tensor, requires_grad=True)
            self.share_rank_B_pool = nn.Parameter(share_rank_B_pool_tensor, requires_grad=True)
        else:
            self.teacher_share_rank_A_pool = None
            self.teacher_share_rank_B_pool = None
            self.share_rank_A_pool = None
            self.share_rank_B_pool = None
        if teacher_part1_size is not None and teacher_part1_size>0:
            teacher_part1_rank_A_pool_tensor = torch.empty(in_features, teacher_part1_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(teacher_part1_rank_A_pool_tensor, mean=0.0, std=bound)
            teacher_part1_rank_B_pool_tensor = torch.zeros(teacher_part1_size, out_features, device=ori_device, dtype=ori_dtype)
            self.teacher_part1_rank_A_pool = nn.Parameter(teacher_part1_rank_A_pool_tensor, requires_grad=False)
            self.teacher_part1_rank_B_pool = nn.Parameter(teacher_part1_rank_B_pool_tensor, requires_grad=False)
        else:
            self.teacher_part1_rank_A_pool = None
            self.teacher_part1_rank_B_pool = None
        if teacher_part2_size is not None and teacher_part2_size>0:
            teacher_part2_rank_A_pool_tensor = torch.empty(in_features, teacher_part2_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(teacher_part2_rank_A_pool_tensor, mean=0.0, std=bound)
            teacher_part2_rank_B_pool_tensor = torch.zeros(teacher_part2_size, out_features, device=ori_device, dtype=ori_dtype)
            self.teacher_part2_rank_A_pool = nn.Parameter(teacher_part2_rank_A_pool_tensor, requires_grad=False)
            self.teacher_part2_rank_B_pool = nn.Parameter(teacher_part2_rank_B_pool_tensor, requires_grad=False)
        else:
            self.teacher_part2_rank_A_pool = None
            self.teacher_part2_rank_B_pool = None
        if train_size is not None and train_size>0:
            train_rank_A_pool_tensor = torch.empty(in_features, train_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(train_rank_A_pool_tensor, mean=0.0, std=bound)   
            train_rank_B_pool_tensor = torch.zeros(train_size, out_features, device=ori_device, dtype=ori_dtype)  
            self.train_rank_A_pool = nn.Parameter(train_rank_A_pool_tensor, requires_grad=True)
            self.train_rank_B_pool = nn.Parameter(train_rank_B_pool_tensor, requires_grad=True)
        else:
            self.train_rank_A_pool = None
            self.train_rank_B_pool = None


    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.lora_dropout is not None:
            x = self.lora_dropout(x)
        state = self.get_state()
        teacher_part1_lora_scores, teacher_part2_lora_scores, train_lora_score = self.get_score(self.pool_name, state)
        if state == "Teacher":
            if self.teacher_share_rank_A_pool is not None and self.teacher_share_rank_B_pool is not None: 
                result += torch.matmul(
                        torch.matmul(x, self.teacher_share_rank_A_pool), self.teacher_share_rank_B_pool
                    ) * self.lora_alpha / self.share_size
        else:
            if self.share_rank_A_pool is not None and self.share_rank_B_pool is not None: 
                result += torch.matmul(
                        torch.matmul(x, self.share_rank_A_pool), self.share_rank_B_pool
                    ) * self.lora_alpha / self.share_size
        if train_lora_score is not None:           
            score_sum = torch.sum(train_lora_score, dim=0)
            _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, train_lora_score.size(1)))
            train_lora_score[:,top_indices] = 1
            result += torch.matmul(
                    (torch.matmul(x, self.train_rank_A_pool[:,top_indices]) * train_lora_score[:,top_indices].unsqueeze(1)), self.train_rank_B_pool[top_indices,:]
                ) * self.lora_alpha / self.top_rank
        elif teacher_part2_lora_scores is not None:
            if self.top_rank <= teacher_part2_lora_scores.size(1):
                score_sum = torch.sum(teacher_part2_lora_scores, dim=0)
                _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, teacher_part2_lora_scores.size(1)))
                teacher_part2_lora_scores[:,top_indices] = 1
                result += torch.matmul(
                        (torch.matmul(x, self.teacher_part2_rank_A_pool[:,top_indices]) * teacher_part2_lora_scores[:,top_indices].unsqueeze(1)), self.teacher_part2_rank_B_pool[top_indices,:]
                    ) * self.lora_alpha / self.top_rank
            else:
                teacher_part2_lora_scores[:,:] = 1
                result += torch.matmul(
                        (torch.matmul(x, self.teacher_part2_rank_A_pool) * teacher_part2_lora_scores.unsqueeze(1)), self.teacher_part2_rank_B_pool
                    ) * self.lora_alpha / self.top_rank
                score_sum = torch.sum(teacher_part1_lora_scores, dim=0)
                _ , top_indices = torch.topk(score_sum, k=self.top_rank - teacher_part2_lora_scores.size(1))
                teacher_part1_lora_scores[:,top_indices] = 1
                result += torch.matmul(
                        (torch.matmul(x, self.teacher_part1_rank_A_pool[:,top_indices]) * teacher_part1_lora_scores[:,top_indices].unsqueeze(1)), self.teacher_part1_rank_B_pool[top_indices,:]
                    ) * self.lora_alpha / self.top_rank
        elif teacher_part1_lora_scores is not None:
            score_sum = torch.sum(teacher_part1_lora_scores, dim=0)
            _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, teacher_part1_lora_scores.size(1)))
            teacher_part1_lora_scores[:,top_indices] = 1
            result += torch.matmul(
                    (torch.matmul(x, self.teacher_part1_rank_A_pool[:,top_indices]) * teacher_part1_lora_scores[:,top_indices].unsqueeze(1)), self.teacher_part1_rank_B_pool[top_indices,:]
                ) * self.lora_alpha / self.top_rank
        return result


class LlmLrpLinear(nn.Linear):
    def __init__(
        self,
        pool_name,
        in_features: int,
        out_features: int,
        share_size, 
        static_size, 
        train_size,
        top_rank,
        freeze_share,
        ori_device, 
        ori_dtype,
        get_score,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.pool_name = pool_name
        self.get_score = get_score
        self.lora_alpha = lora_alpha
        self.share_size = share_size
        self.static_size = static_size
        self.train_size = train_size
        self.top_rank = top_rank
        self.weight.requires_grad = False
        nn.Linear.reset_parameters(self)
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = None
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)

        if share_size is not None and share_size>0:
            share_rank_A_pool_tensor = torch.empty(in_features, share_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(share_rank_A_pool_tensor, mean=0.0, std=bound)
            share_rank_B_pool_tensor = torch.zeros(share_size, out_features, device=ori_device, dtype=ori_dtype)
            if freeze_share:
                self.share_rank_A_pool = nn.Parameter(share_rank_A_pool_tensor, requires_grad=False)
                self.share_rank_B_pool = nn.Parameter(share_rank_B_pool_tensor, requires_grad=False)
            else:
                self.share_rank_A_pool = nn.Parameter(share_rank_A_pool_tensor, requires_grad=True)
                self.share_rank_B_pool = nn.Parameter(share_rank_B_pool_tensor, requires_grad=True)
        else:
            self.share_rank_A_pool = None
            self.share_rank_B_pool = None
        if static_size is not None and static_size>0:
            static_rank_A_pool_tensor = torch.empty(in_features, static_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(static_rank_A_pool_tensor, mean=0.0, std=bound)
            static_rank_B_pool_tensor = torch.zeros(static_size, out_features, device=ori_device, dtype=ori_dtype)
            self.static_rank_A_pool = nn.Parameter(static_rank_A_pool_tensor, requires_grad=False)
            self.static_rank_B_pool = nn.Parameter(static_rank_B_pool_tensor, requires_grad=False)
        else:
            self.static_rank_A_pool = None
            self.static_rank_B_pool = None
        if train_size is not None and train_size>0:
            train_rank_A_pool_tensor = torch.empty(in_features, train_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(train_rank_A_pool_tensor, mean=0.0, std=bound)   
            train_rank_B_pool_tensor = torch.zeros(train_size, out_features, device=ori_device, dtype=ori_dtype)  
            self.train_rank_A_pool = nn.Parameter(train_rank_A_pool_tensor, requires_grad=True)
            self.train_rank_B_pool = nn.Parameter(train_rank_B_pool_tensor, requires_grad=True)
        else:
            self.train_rank_A_pool = None
            self.train_rank_B_pool = None


    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.lora_dropout is not None:
            x = self.lora_dropout(x)
        static_lora_score, train_lora_score = self.get_score(self.pool_name)     
        if self.share_rank_A_pool is not None and self.share_rank_B_pool is not None: 
            result += torch.matmul(
                    torch.matmul(x, self.share_rank_A_pool), self.share_rank_B_pool
                ) * self.lora_alpha / self.share_size
        if self.top_rank is None:
            if self.static_rank_A_pool is not None and self.static_rank_B_pool is not None and static_lora_score is not None and static_lora_score.shape[0]==x.shape[0]: 
                result += torch.matmul(
                        (torch.matmul(x, self.static_rank_A_pool) * static_lora_score.unsqueeze(1)), self.static_rank_B_pool
                    )
            elif self.train_rank_A_pool is not None and self.train_rank_B_pool is not None and train_lora_score is not None and train_lora_score.shape[0]==x.shape[0]: 
                result += torch.matmul(
                        (torch.matmul(x, self.train_rank_A_pool) * train_lora_score.unsqueeze(1)), self.train_rank_B_pool
                    )
        else:        
            if train_lora_score is not None:
                if self.top_rank <= train_lora_score.size(1):
                    score_sum = torch.sum(train_lora_score, dim=0)
                    _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, train_lora_score.size(1)))
                    train_lora_score[:,top_indices] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.train_rank_A_pool[:,top_indices]) * train_lora_score[:,top_indices].unsqueeze(1)), self.train_rank_B_pool[top_indices,:]
                        ) * self.lora_alpha / self.top_rank
                else:
                    train_lora_score[:,:] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.train_rank_A_pool) * train_lora_score.unsqueeze(1)), self.train_rank_B_pool
                        ) * self.lora_alpha / self.top_rank
                    score_sum = torch.sum(static_lora_score, dim=0)
                    _ , top_indices = torch.topk(score_sum, k=self.top_rank - train_lora_score.size(1))
                    static_lora_score[:,top_indices] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.static_rank_A_pool[:,top_indices]) * static_lora_score[:,top_indices].unsqueeze(1)), self.static_rank_B_pool[top_indices,:]
                        ) * self.lora_alpha / self.top_rank
            elif static_lora_score is not None:
                score_sum = torch.sum(static_lora_score, dim=0)
                _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, static_lora_score.size(1)))
                static_lora_score[:,top_indices] = 1
                result += torch.matmul(
                        (torch.matmul(x, self.static_rank_A_pool[:,top_indices]) * static_lora_score[:,top_indices].unsqueeze(1)), self.static_rank_B_pool[top_indices,:]
                    ) * self.lora_alpha / self.top_rank
        return result
    
    
class mmLrpLinear(nn.Linear):
    def __init__(
        self,
        pool_name,
        in_features: int,
        out_features: int,
        share_size,
        static_size, 
        train_size,
        top_rank,
        ori_device, 
        ori_dtype,
        get_score,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.pool_name = pool_name
        self.get_score = get_score
        self.lora_alpha = lora_alpha
        self.share_size = share_size
        self.static_size = static_size
        self.train_size = train_size
        self.top_rank = top_rank
        self.weight.requires_grad = False
        nn.Linear.reset_parameters(self)
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = None
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        if share_size is not None and share_size>0:
            share_rank_A_pool_tensor = torch.empty(in_features, share_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(share_rank_A_pool_tensor, mean=0.0, std=bound)
            share_rank_B_pool_tensor = torch.zeros(share_size, out_features, device=ori_device, dtype=ori_dtype)
            self.share_rank_A_pool = nn.Parameter(share_rank_A_pool_tensor, requires_grad=False)
            self.share_rank_B_pool = nn.Parameter(share_rank_B_pool_tensor, requires_grad=False)
        else:
            self.share_rank_A_pool = None
            self.share_rank_B_pool = None
        if static_size is not None and static_size>0:
            static_rank_A_pool_tensor = torch.empty(in_features, static_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(static_rank_A_pool_tensor, mean=0.0, std=bound)
            static_rank_B_pool_tensor = torch.zeros(static_size, out_features, device=ori_device, dtype=ori_dtype)
            self.static_rank_A_pool = nn.Parameter(static_rank_A_pool_tensor, requires_grad=False)
            self.static_rank_B_pool = nn.Parameter(static_rank_B_pool_tensor, requires_grad=False)
        else:
            self.static_rank_A_pool = None
            self.static_rank_B_pool = None
        if train_size is not None and train_size>0:
            train_rank_A_pool_tensor = torch.empty(in_features, train_size,
                            device=ori_device, dtype=ori_dtype)
            nn.init.normal_(train_rank_A_pool_tensor, mean=0.0, std=bound)   
            train_rank_B_pool_tensor = torch.zeros(train_size, out_features, device=ori_device, dtype=ori_dtype)  
            self.train_rank_A_pool = nn.Parameter(train_rank_A_pool_tensor, requires_grad=True)
            self.train_rank_B_pool = nn.Parameter(train_rank_B_pool_tensor, requires_grad=True)
        else:
            self.train_rank_A_pool = None
            self.train_rank_B_pool = None


    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        if self.lora_dropout is not None:
            x = self.lora_dropout(x)
        static_lora_score, train_lora_score = self.get_score(self.pool_name)
        if self.share_rank_A_pool is not None and self.share_rank_B_pool is not None: 
            result += torch.matmul(
                    torch.matmul(x, self.share_rank_A_pool), self.share_rank_B_pool
                ) * self.lora_alpha / self.share_size     
        if self.top_rank is None:
            if self.static_rank_A_pool is not None and self.static_rank_B_pool is not None and static_lora_score is not None and static_lora_score.shape[0]==x.shape[0]: 
                result += torch.matmul(
                        (torch.matmul(x, self.static_rank_A_pool) * static_lora_score.unsqueeze(1)), self.static_rank_B_pool
                    )
            elif self.train_rank_A_pool is not None and self.train_rank_B_pool is not None and train_lora_score is not None and train_lora_score.shape[0]==x.shape[0]: 
                result += torch.matmul(
                        (torch.matmul(x, self.train_rank_A_pool) * train_lora_score.unsqueeze(1)), self.train_rank_B_pool
                    )
        else:        
            if train_lora_score is not None:
                if self.top_rank <= train_lora_score.size(1):
                    score_sum = torch.sum(train_lora_score, dim=0)
                    _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, train_lora_score.size(1)))
                    train_lora_score[:,top_indices] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.train_rank_A_pool[:,top_indices]) * train_lora_score[:,top_indices].unsqueeze(1)), self.train_rank_B_pool[top_indices,:]
                        ) * self.lora_alpha / self.top_rank
                else:
                    train_lora_score[:,:] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.train_rank_A_pool) * train_lora_score.unsqueeze(1)), self.train_rank_B_pool
                        ) * self.lora_alpha / self.top_rank
                    score_sum = torch.sum(static_lora_score, dim=0)
                    _ , top_indices = torch.topk(score_sum, k=self.top_rank - train_lora_score.size(1))
                    static_lora_score[:,top_indices] = 1
                    result += torch.matmul(
                            (torch.matmul(x, self.static_rank_A_pool[:,top_indices]) * static_lora_score[:,top_indices].unsqueeze(1)), self.static_rank_B_pool[top_indices,:]
                        ) * self.lora_alpha / self.top_rank
            elif static_lora_score is not None:
                score_sum = torch.sum(static_lora_score, dim=0)
                _ , top_indices = torch.topk(score_sum, k=min(self.top_rank, static_lora_score.size(1)))
                static_lora_score[:,top_indices] = 1
                result += torch.matmul(
                        (torch.matmul(x, self.static_rank_A_pool[:,top_indices]) * static_lora_score[:,top_indices].unsqueeze(1)), self.static_rank_B_pool[top_indices,:]
                    ) * self.lora_alpha / self.top_rank
        return result