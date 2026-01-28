import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from lrp_llava_hf.model.utils import LrpConfig, _get_submodules, LlmLrpTSLinear, mmLrpLinear, get_llm_query
import copy
import math
import torch
import random




class LrpTSModel(nn.Module):
    def __init__(self, base_model, lrp_config: Optional[LrpConfig] = None, A_loaded = None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.lrp_config = lrp_config
        self.teacher_part1_keys_vit = nn.ParameterDict()
        self.teacher_part2_keys_vit = nn.ParameterDict()
        self.train_keys_vit = nn.ParameterDict()
        self.static_keys_vit = nn.ParameterDict()
        self.teacher_part1_keys_llm = nn.ParameterDict()
        self.teacher_part2_keys_llm = nn.ParameterDict()
        self.train_keys_llm = nn.ParameterDict()
        self.static_keys_llm = nn.ParameterDict()
        self.vit_hidden_size = lrp_config.vit_hidden_size
        self.llm_hidden_size = lrp_config.llm_hidden_size
        self.top_rank = self.lrp_config.llm_top_rank
        self.part = "part1"
        self.task_choose = ""
        if lrp_config is not None:
            self.ori_device = lrp_config.device
            self.ori_dtype = lrp_config.dtype
        self.register_buffer('llm_query', None)
        self.register_buffer('vit_query', None)
        self.register_buffer('no_image', None)
        self.state = "Teacher"
        self.task_num = self.lrp_config.task_num
        self.soft = torch.nn.Softmax(dim=2)
        if self.task_num > 0:
            task_llm_query_tensor = torch.zeros(self.task_num, self.llm_hidden_size, device=self.ori_device, dtype=self.ori_dtype)
            task_vit_query_tensor = torch.zeros(self.task_num, self.vit_hidden_size, device=self.ori_device, dtype=self.ori_dtype)
            self.task_llm_query_pool = nn.Parameter(task_llm_query_tensor, requires_grad=False)
            self.task_vit_query_pool = nn.Parameter(task_vit_query_tensor, requires_grad=False)
        else:
            self.task_llm_query_pool = None
            self.task_vit_query_pool = None
        if lrp_config.use_loss_probability:
            probability = torch.ones(self.task_num).to(device=self.ori_device, dtype=torch.float32)
            self.task_probability = probability
        else:
            self.task_probability = None
        if lrp_config.llm_share_rank_size is not None and lrp_config.llm_share_rank_size > 0:
            self.share = True
        else:
            self.share = False
        self._load_pools()

    @torch.no_grad()
    def change_probability(self, loss):
        if self.task_probability is None:
            return
        self.task_probability[self.task_choose] = torch.sqrt(torch.clamp(loss.to(dtype=torch.float32), min=0.0) + 0.0001)
        print(self.task_choose, self.task_probability)


    def get_state(self):
        return self.state
    
    def init_key_value(self, skip_interval):
        # mode = "order_match"
        mode = "meantime_match"
        k = self.vit_hidden_size / self.llm_hidden_size
        for pool_name in self.lrp_config.llm_target_modules.keys():
            if self.teacher_part1_keys_llm[pool_name] is not None:
                teacher_part1_indices = torch.arange(self.teacher_part1_keys_llm[pool_name].shape[0])
                teacher_part2_indices = torch.arange(self.teacher_part2_keys_llm[pool_name].shape[0])
                teacher_part1_mask = (teacher_part1_indices + 1) % skip_interval != 0
                teacher_part2_mask = (teacher_part2_indices + 1) % skip_interval != 0
                processed_A = self.teacher_part1_keys_llm[pool_name][teacher_part1_mask]
                processed_B = self.teacher_part2_keys_llm[pool_name][teacher_part2_mask]
                C_llm = F.normalize(torch.cat((processed_A, processed_B), dim=0).clone().detach(), p=2, dim=1)
                processed_A = self.teacher_part1_keys_vit[pool_name][teacher_part1_mask]
                processed_B = self.teacher_part2_keys_vit[pool_name][teacher_part2_mask]
                C_vit = F.normalize(torch.cat((processed_A, processed_B), dim=0).clone().detach(), p=2, dim=1)
                if mode == "meantime_match":
                    score = torch.einsum('qd,cd->qc', self.task_llm_query_pool, C_llm) + k * torch.einsum('qd,cd->qc', self.task_vit_query_pool, C_vit)
                    _, indices = torch.topk(score, k=self.lrp_config.llm_top_rank, dim=1)
                    for i in range(indices.shape[0]):
                        C_llm[indices[i]] += self.task_llm_query_pool[i]
                        C_vit[indices[i]] += self.task_vit_query_pool[i]
                elif mode == "order_match":
                    alpha = torch.ones(C_llm.shape[0]).to(device = C_llm.device, dtype = C_llm.dtype)
                    for i in range(self.task_llm_query_pool.shape[0]):
                        score = torch.einsum('d,cd->c', self.task_llm_query_pool[i], C_llm) + k * torch.einsum('d,cd->c', self.task_vit_query_pool[i], C_vit)
                        _, indices = torch.topk(score, k=self.lrp_config.llm_top_rank, dim=0)
                        C_llm[indices] = C_llm[indices] * (1 - 1 / (1 + alpha[indices].unsqueeze(-1))) + self.task_llm_query_pool[i] * (1 / (1 + alpha[indices].unsqueeze(-1)))
                        C_vit[indices] = C_vit[indices] * (1 - 1 / (1 + alpha[indices].unsqueeze(-1))) + self.task_vit_query_pool[i] * (1 / (1 + alpha[indices].unsqueeze(-1)))
                        alpha[indices] += 1
                C_llm = F.normalize(C_llm, p=2, dim=1)
                C_vit = F.normalize(C_vit, p=2, dim=1)
                self.train_keys_llm[pool_name] = nn.Parameter(C_llm, requires_grad=False)
                self.train_keys_vit[pool_name] = nn.Parameter(C_vit, requires_grad=False)
                for target_name in self.lrp_config.llm_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    processed_A = target.teacher_part1_rank_A_pool[:,teacher_part1_mask]
                    processed_B = target.teacher_part2_rank_A_pool[:,teacher_part2_mask]
                    C = torch.cat((processed_A, processed_B), dim=1).clone().detach()
                    target.train_rank_A_pool = nn.Parameter(C, requires_grad=True)
                    processed_A = target.teacher_part1_rank_B_pool[teacher_part1_mask]
                    processed_B = target.teacher_part2_rank_B_pool[teacher_part2_mask]
                    C = torch.cat((processed_A, processed_B), dim=0).clone().detach()
                    target.train_rank_B_pool = nn.Parameter(C, requires_grad=True)
                    if self.share:
                        C = target.teacher_share_rank_A_pool.clone().detach()
                        target.share_rank_A_pool = nn.Parameter(C, requires_grad=True)
                        C = target.teacher_share_rank_B_pool.clone().detach()
                        target.share_rank_B_pool = nn.Parameter(C, requires_grad=True)
            else:
                teacher_part2_indices = torch.arange(self.teacher_part2_keys_llm[pool_name].shape[0])
                teacher_part2_mask = (teacher_part2_indices + 1) % skip_interval != 0
                processed_B = self.teacher_part2_keys_llm[pool_name][teacher_part2_mask]
                C_llm = F.normalize(processed_B.clone().detach(), p=2, dim=1)
                processed_B = self.teacher_part2_keys_vit[pool_name][teacher_part2_mask]
                C_vit = F.normalize(processed_B.clone().detach(), p=2, dim=1)
                if mode == "meantime_match":
                    score = torch.einsum('qd,cd->qc', self.task_llm_query_pool, C_llm) + k * torch.einsum('qd,cd->qc', self.task_vit_query_pool, C_vit)
                    _, indices = torch.topk(score, k=self.lrp_config.llm_top_rank, dim=1)
                    for i in range(indices.shape[0]):
                        C_llm[indices[i]] += self.task_llm_query_pool[i]
                        C_vit[indices[i]] += self.task_vit_query_pool[i]
                elif mode == "order_match":
                    alpha = torch.ones(C_llm.shape[0]).to(device = C_llm.device, dtype = C_llm.dtype)
                    for i in range(self.task_llm_query_pool.shape[0]):
                        score = torch.einsum('d,cd->c', self.task_llm_query_pool[i], C_llm) + k * torch.einsum('d,cd->c', self.task_vit_query_pool[i], C_vit)
                        _, indices = torch.topk(score, k=self.lrp_config.llm_top_rank, dim=0)
                        C_llm[indices] = C_llm[indices] * (1 - 1 / (1 + alpha[indices].unsqueeze(-1))) + self.task_llm_query_pool[i] * (1 / (1 + alpha[indices].unsqueeze(-1)))
                        C_vit[indices] = C_vit[indices] * (1 - 1 / (1 + alpha[indices].unsqueeze(-1))) + self.task_vit_query_pool[i] * (1 / (1 + alpha[indices].unsqueeze(-1)))
                        alpha[indices] += 1
                
                C_llm = F.normalize(C_llm, p=2, dim=1)
                C_vit = F.normalize(C_vit, p=2, dim=1)
                self.train_keys_llm[pool_name] = nn.Parameter(C_llm, requires_grad=False)
                self.train_keys_vit[pool_name] = nn.Parameter(C_vit, requires_grad=False)
                for target_name in self.lrp_config.llm_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    processed_B = target.teacher_part2_rank_A_pool[:,teacher_part2_mask]
                    C = processed_B.clone().detach()
                    target.train_rank_A_pool = nn.Parameter(C, requires_grad=True)
                    processed_B = target.teacher_part2_rank_B_pool[teacher_part2_mask]
                    C = processed_B.clone().detach()
                    target.train_rank_B_pool = nn.Parameter(C, requires_grad=True)
                    if self.share:
                        C = target.teacher_share_rank_A_pool.clone().detach()
                        target.share_rank_A_pool = nn.Parameter(C, requires_grad=True)
                        C = target.teacher_share_rank_B_pool.clone().detach()
                        target.share_rank_B_pool = nn.Parameter(C, requires_grad=True)

            

    def _load_pools(self):
        llm_kwargs = {
            "share_size": self.lrp_config.llm_share_rank_size, 
            "teacher_part1_size": self.lrp_config.llm_teacher_part1_rank_size,
            "teacher_part2_size": self.lrp_config.llm_teacher_part2_rank_size,
            "train_size": self.lrp_config.llm_train_rank_size,
            "top_rank": self.lrp_config.llm_top_rank,
            "freeze_share": self.lrp_config.llm_freeze_share,
            "ori_device": self.ori_device, 
            "ori_dtype": self.ori_dtype, 
            "get_state": self.get_state,
            "get_score": self.get_score,
            "lora_alpha": self.lrp_config.llm_lora_alpha,
            "lora_dropout": self.lrp_config.llm_lora_dropout,
        }
        mm_kwargs = {
            "share_size": self.lrp_config.mm_share_rank_size, 
            "static_size": self.lrp_config.mm_static_rank_size,
            "train_size": self.lrp_config.mm_train_rank_size,
            "top_rank": self.lrp_config.mm_top_rank,
            "ori_device": self.ori_device, 
            "ori_dtype": self.ori_dtype, 
            "get_score": self.get_static_score,
            "lora_alpha": self.lrp_config.mm_lora_alpha,
            "lora_dropout": self.lrp_config.mm_lora_dropout,
        }
        for pool_name in self.lrp_config.llm_target_modules.keys():
            self.create_key_pool("llm", "llm", pool_name)
            self.create_key_pool("llm", "vit", pool_name)
            for target_name in self.lrp_config.llm_target_modules[pool_name]:
                parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                bias = target.bias is not None
                llm_kwargs['bias'] = bias
                in_features, out_features = target.in_features, target.out_features
                new_module = LlmLrpTSLinear(pool_name, in_features, out_features, **llm_kwargs)                   
                self._replace_module(parent, target_name_, new_module, target)
        if self.lrp_config.mm_target_modules is not None:
            for pool_name in self.lrp_config.mm_target_modules.keys():
                self.create_static_key_pool("mm", "llm", pool_name)
                self.create_static_key_pool("mm", "vit", pool_name)
                for target_name in self.lrp_config.mm_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    bias = target.bias is not None
                    mm_kwargs['bias'] = bias
                    in_features, out_features = target.in_features, target.out_features
                    new_module = mmLrpLinear(pool_name, in_features, out_features, **mm_kwargs)                   
                    self._replace_module(parent, target_name_, new_module, target)

    def create_key_pool(self, module_type: str, key_type: str, pool_name: str):
        if module_type == 'llm':
            teacher_part1_size = self.lrp_config.llm_teacher_part1_rank_size
            teacher_part2_size = self.lrp_config.llm_teacher_part2_rank_size
            train_size = self.lrp_config.llm_train_rank_size
        elif module_type == 'mm': 
            static_size = self.lrp_config.mm_static_rank_size
            train_size = self.lrp_config.mm_train_rank_size
        else:  # vit
            teacher_part1_size = self.lrp_config.vit_teacher_part1_prompt_size
            teacher_part2_size = self.lrp_config.vit_teacher_part2_prompt_size
            train_size = self.lrp_config.vit_train_prompt_size
        if key_type == 'llm':
            k_hidden_size = self.llm_hidden_size
        else:
            k_hidden_size = self.vit_hidden_size
        if teacher_part1_size is not None and teacher_part1_size > 0:
            teacher_part1_key_pool = torch.randn(teacher_part1_size, k_hidden_size,
                            device=self.ori_device, dtype=self.ori_dtype)
            teacher_part1_key_pool = F.normalize(teacher_part1_key_pool, p=2, dim=1)
            teacher_part1_param = nn.Parameter(teacher_part1_key_pool, requires_grad=False)
            if key_type == 'llm':
                self.teacher_part1_keys_llm[pool_name] = teacher_part1_param
            else:
                self.teacher_part1_keys_vit[pool_name] = teacher_part1_param
        else:
            if key_type == 'llm':
                self.teacher_part1_keys_llm[pool_name] = None
            else:
                self.teacher_part1_keys_vit[pool_name] = None
        if teacher_part2_size is not None and teacher_part2_size > 0:
            teacher_part2_key_pool = torch.randn(teacher_part2_size, k_hidden_size,
                            device=self.ori_device, dtype=self.ori_dtype)
            teacher_part2_key_pool = F.normalize(teacher_part2_key_pool, p=2, dim=1)
            teacher_part2_param = nn.Parameter(teacher_part2_key_pool, requires_grad=False)
            if key_type == 'llm':
                self.teacher_part2_keys_llm[pool_name] = teacher_part2_param
            else:
                self.teacher_part2_keys_vit[pool_name] = teacher_part2_param
        else:
            if key_type == 'llm':
                self.teacher_part2_keys_llm[pool_name] = None
            else:
                self.teacher_part2_keys_vit[pool_name] = None
        if train_size is not None and train_size > 0:
            train_key_pool = torch.randn(train_size, k_hidden_size,
                            device=self.ori_device, dtype=self.ori_dtype)
            train_key_pool = F.normalize(train_key_pool, p=2, dim=1)
            train_param = nn.Parameter(train_key_pool, requires_grad=False)
            if key_type == 'llm':
                self.train_keys_llm[pool_name] = train_param
            else:
                self.train_keys_vit[pool_name] = train_param
        else:
            if key_type == 'llm':
                self.train_keys_llm[pool_name] = None
            else:
                self.train_keys_vit[pool_name] = None


    def get_score(self, pool_name, state):
        if self.llm_query is None and self.vit_query is None:
            return None, None
        k = self.vit_hidden_size / self.llm_hidden_size
        if state == "Teacher":
            if self.teacher_part1_keys_llm[pool_name] is not None:
                attn_teacher_part1_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.teacher_part1_keys_llm[pool_name], p=2, dim=1))
                attn_teacher_part1_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.teacher_part1_keys_vit[pool_name], p=2, dim=1))
                attn_teacher_part2_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.teacher_part2_keys_llm[pool_name], p=2, dim=1))
                attn_teacher_part2_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.teacher_part2_keys_vit[pool_name], p=2, dim=1))
                teacher_part1_score = attn_teacher_part1_llm + k * attn_teacher_part1_vit
                teacher_part2_score = attn_teacher_part2_llm + k * attn_teacher_part2_vit
                teacher_part1_score_max = torch.max(torch.sum(teacher_part1_score, dim=0))
                teacher_part2_score_max = torch.max(torch.sum(teacher_part2_score, dim=0))
                if self.part == "part1":
                    return teacher_part1_score, None, None
                else:
                    return teacher_part1_score, teacher_part2_score, None
            else:
                attn_teacher_part2_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.teacher_part2_keys_llm[pool_name], p=2, dim=1))
                attn_teacher_part2_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.teacher_part2_keys_vit[pool_name], p=2, dim=1))
                teacher_part2_score = attn_teacher_part2_llm + k * attn_teacher_part2_vit
                return None, teacher_part2_score, None
        elif state == "Student":
            attn_train_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.train_keys_llm[pool_name], p=2, dim=1)) if self.train_keys_llm[pool_name] is not None else None
            attn_train_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.train_keys_vit[pool_name], p=2, dim=1)) if self.train_keys_vit[pool_name] is not None else None 
            train_score = attn_train_llm + k * attn_train_vit
            return None, None, train_score


    def create_static_key_pool(self, module_type: str, key_type: str, pool_name: str):
        static_size = self.lrp_config.mm_static_rank_size
        if key_type == 'llm':
            k_hidden_size = self.llm_hidden_size
        else:
            k_hidden_size = self.vit_hidden_size
        if static_size is not None and static_size > 0:
            static_key_pool = torch.randn(static_size, k_hidden_size,
                            device=self.ori_device, dtype=self.ori_dtype)
            static_key_pool = F.normalize(static_key_pool, p=2, dim=1)
            static_param = nn.Parameter(static_key_pool, requires_grad=False)
            if key_type == 'llm':
                self.static_keys_llm[pool_name] = static_param
            else:
                self.static_keys_vit[pool_name] = static_param
        else:
            if key_type == 'llm':
                self.static_keys_llm[pool_name] = None
            else:
                self.static_keys_vit[pool_name] = None


    def get_static_score(self, pool_name):
        if self.llm_query is None and self.vit_query is None:
            return None, None
        k = self.vit_hidden_size / self.llm_hidden_size
        attn_static_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.static_keys_llm[pool_name], p=2, dim=1))
        attn_static_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.static_keys_vit[pool_name], p=2, dim=1))
        static_score = attn_static_llm + k * attn_static_vit
        return static_score, None

    
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
        
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.use_gradient_checkpointing = True
        if hasattr(self, "base_model"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
    def gradient_checkpointing_disable(self):
        self.use_gradient_checkpointing = False
        if hasattr(self, "base_model"):
            self.base_model.gradient_checkpointing_disable()
            
    @torch.no_grad()
    def teacher_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_ids: Optional[torch.FloatTensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        has_image: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.vit_query = None
        self.llm_query = None
        batch_size = input_ids.shape[0]
        id_max = self.task_llm_query_pool.shape[0] - 1
        self.no_image = torch.tensor([i for i, item in enumerate(has_image) if item == 0]).to(dtype=torch.int)
        if self.task_probability is not None:
            probability = self.task_probability / self.task_probability.sum()
            task_identify = torch.multinomial(probability, num_samples=1).item()
        else:
            task_identify = random.randint(0, id_max)
        if task_identify == id_max:
            self.part = "part2"
        else:
            self.part = "part1"
        self.task_choose = task_identify
        vit_query_single = self.task_vit_query_pool[self.task_choose].detach().clone()
        llm_query_single = self.task_llm_query_pool[self.task_choose].detach().clone()
        vit_query = vit_query_single.expand(batch_size, -1)
        llm_query = llm_query_single.expand(batch_size, -1)
        vit_query[self.no_image,:] = 0
        self.llm_query = llm_query.detach()
        self.vit_query = vit_query.detach()
        self.state = "Teacher"

        if images is not None:
            teacher_output = self.base_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                pixel_values = images,
                return_dict = return_dict,
                **kwargs
            )
        else:
            teacher_output = self.base_model.language_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                return_dict = return_dict,
                **kwargs
            )
        teacher_logits = teacher_output['logits']
        return {"logits": teacher_logits}  
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_ids: Optional[torch.FloatTensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        has_image: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
        random_choose: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.state = "Student"
        if images is not None:
            student_output = self.base_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                pixel_values = images,
                return_dict = return_dict,
                **kwargs
                )
        else:
            student_output = self.base_model.language_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                return_dict = return_dict,
                **kwargs
            )
        student_logits = student_output['logits']
        return {"logits": student_logits}
   

    @classmethod
    def from_pretrained(cls, base_model, pretrained_model_name_or_path, **kwargs):
        mode = kwargs.pop('mode', 'train')
        if mode == 'train':
            config_name = 'lrp_config.json'
        elif mode == 'generate':
            config_name = 'lrp_generate_config.json'
        lrp_config = LrpConfig.from_pretrained(pretrained_model_name_or_path, config_name)
        lrp_config.device = base_model.model.device
        lrp_config.dtype = base_model.model.dtype
        return cls(base_model, lrp_config)