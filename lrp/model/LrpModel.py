import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from lrp.model.utils import LrpConfig, _get_submodules, LlmLrpLinear, VitLrpLayerNorm, mmLrpLinear, get_llm_query
import copy
import math
import torch

class LrpModel(nn.Module):
    def __init__(self, base_model, lrp_config: Optional[LrpConfig] = None, A_loaded = None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.lrp_config = lrp_config
        self.static_keys_vit = nn.ParameterDict()
        self.train_keys_vit = nn.ParameterDict()
        self.static_keys_llm = nn.ParameterDict()
        self.train_keys_llm = nn.ParameterDict()
        self.vit_hidden_size = lrp_config.vit_hidden_size
        self.llm_hidden_size = lrp_config.llm_hidden_size
        if lrp_config is not None:
            self.ori_device = lrp_config.device
            self.ori_dtype = lrp_config.dtype
        self.register_buffer('llm_query', None)
        self.register_buffer('vit_query', None)
        self.register_buffer('no_image', None)
        self.task_num = self.lrp_config.task_num
        if self.task_num is not None and self.task_num > 0:
            task_llm_query_tensor = torch.zeros(self.task_num, self.llm_hidden_size, device=self.ori_device, dtype=self.ori_dtype)
            task_vit_query_tensor = torch.zeros(self.task_num, self.vit_hidden_size, device=self.ori_device, dtype=self.ori_dtype)
            self.task_llm_query_pool = nn.Parameter(task_llm_query_tensor, requires_grad=False)
            self.task_vit_query_pool = nn.Parameter(task_vit_query_tensor, requires_grad=False)
        else:
            self.task_llm_query_pool = None
            self.task_vit_query_pool = None
        self.generate_part = "new"
        self._load_pools()

    def init_key(self, data):
        for pool_name in self.train_keys_llm.keys():
            key_data = data["llm"].to(device=self.ori_device, dtype=self.ori_dtype)
            key = torch.zeros(self.train_keys_llm[pool_name].shape[0], self.train_keys_llm[pool_name].shape[1], device=self.ori_device, dtype=self.ori_dtype)
            for i in range(key.shape[0]):
                key[i,:] = key_data[0,:].clone().detach()
            self.train_keys_llm[pool_name] = nn.Parameter(key, requires_grad=True)
        for pool_name in self.train_keys_vit.keys():
            key_data = data["vit"].to(device=self.ori_device, dtype=self.ori_dtype)
            key = torch.zeros(self.train_keys_vit[pool_name].shape[0], self.train_keys_vit[pool_name].shape[1], device=self.ori_device, dtype=self.ori_dtype)
            for i in range(key.shape[0]):
                key[i,:] = key_data[0,:].clone().detach()
            self.train_keys_vit[pool_name] = nn.Parameter(key, requires_grad=True)
        
    def init_value(self, init_near):
        k = self.vit_hidden_size / self.llm_hidden_size
        for pool_name in self.lrp_config.llm_target_modules.keys():
            key_train_llm = self.train_keys_llm[pool_name][0,:].clone().detach()
            key_train_vit = self.train_keys_vit[pool_name][0,:].clone().detach()
            score_llm = torch.einsum('d,kd->k', key_train_llm, self.static_keys_llm[pool_name])
            score_vit = torch.einsum('d,kd->k', key_train_vit, self.static_keys_vit[pool_name])
            score = score_llm + k * score_vit
            _, indices = torch.topk(score, self.lrp_config.llm_train_rank_size)
            for target_name in self.lrp_config.llm_target_modules[pool_name]:
                parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                if init_near:
                    train_rank_A_pool = target.static_rank_A_pool[:,-self.lrp_config.llm_train_rank_size:].clone().detach()
                    train_rank_B_pool = target.static_rank_B_pool[-self.lrp_config.llm_train_rank_size:].clone().detach()
                    target.train_rank_A_pool = nn.Parameter(train_rank_A_pool, requires_grad=True)
                    target.train_rank_B_pool = nn.Parameter(train_rank_B_pool, requires_grad=True)
                else:
                    train_rank_A_pool = target.static_rank_A_pool[:,indices].clone().detach()
                    train_rank_B_pool = target.static_rank_B_pool[indices].clone().detach()
                    target.train_rank_A_pool = nn.Parameter(train_rank_A_pool, requires_grad=True)
                    target.train_rank_B_pool = nn.Parameter(train_rank_B_pool, requires_grad=True)
        if self.lrp_config.mm_target_modules:
            for pool_name in self.lrp_config.mm_target_modules.keys():
                key_train_llm = self.train_keys_llm[pool_name][0,:].clone().detach()
                key_train_vit = self.train_keys_vit[pool_name][0,:].clone().detach()
                score_llm = torch.einsum('d,kd->k', key_train_llm, self.static_keys_llm[pool_name])
                score_vit = torch.einsum('d,kd->k', key_train_vit, self.static_keys_vit[pool_name])
                score = score_llm + k * score_vit
                _, indices = torch.topk(score, self.lrp_config.mm_train_rank_size)
                for target_name in self.lrp_config.mm_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    if init_near:
                        train_rank_A_pool = target.static_rank_A_pool[:,-self.lrp_config.mm_train_rank_size:].clone().detach()
                        train_rank_B_pool = target.static_rank_B_pool[-self.lrp_config.mm_train_rank_size:].clone().detach()
                        target.train_rank_A_pool = nn.Parameter(train_rank_A_pool, requires_grad=True)
                        target.train_rank_B_pool = nn.Parameter(train_rank_B_pool, requires_grad=True)
                    else:
                        train_rank_A_pool = target.static_rank_A_pool[:,indices].clone().detach()
                        train_rank_B_pool = target.static_rank_B_pool[indices].clone().detach()
                        target.train_rank_A_pool = nn.Parameter(train_rank_A_pool, requires_grad=True)
                        target.train_rank_B_pool = nn.Parameter(train_rank_B_pool, requires_grad=True)
        for pool_name in self.lrp_config.vit_target_modules.keys():
            key_train_llm = self.train_keys_llm[pool_name][0,:].clone().detach()
            key_train_vit = self.train_keys_vit[pool_name][0,:].clone().detach()
            score_llm = torch.einsum('d,kd->k', key_train_llm, self.static_keys_llm[pool_name])
            score_vit = torch.einsum('d,kd->k', key_train_vit, self.static_keys_vit[pool_name])
            score = score_llm + k * score_vit
            _, indices = torch.topk(score, self.lrp_config.vit_train_prompt_size)
            for target_name in self.lrp_config.vit_target_modules[pool_name]:
                parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                if init_near:
                    train_prompt_pool = target.static_prompt_pool[-self.lrp_config.vit_train_prompt_size:,:,:].clone().detach()
                    target.train_prompt_pool = nn.Parameter(train_prompt_pool, requires_grad=True)
                else:
                    train_prompt_pool = target.static_prompt_pool[indices,:,:].clone().detach()
                    target.train_prompt_pool = nn.Parameter(train_prompt_pool, requires_grad=True)
            
            
    def unit_orthogonalization(self):
        for pool_name in self.train_keys_vit.keys():
            if self.train_keys_vit[pool_name] is not None:
                current_tensor = self.train_keys_vit[pool_name].data.clone()
                vector_num, vector_dim = current_tensor.shape
                for i in range(vector_num):
                    if i == 0:
                        new_vector = current_tensor[0, :]
                        new_vector = F.normalize(new_vector, dim=0)
                        current_tensor[0, :] = new_vector
                        continue
                    existing = current_tensor[0:i,:]
                    if i < vector_dim:
                        new_vector = current_tensor[i, :]
                        for existing_vec in existing:
                            projection = (new_vector @ existing_vec) / (existing_vec @ existing_vec)
                            new_vector = new_vector - projection * existing_vec
                        new_vector = F.normalize(new_vector, dim=0)
                    else:
                        new_vector = current_tensor[i, :]
                        new_vector = F.normalize(new_vector, dim=0)
                    if torch.norm(new_vector) < 1e-3:
                        while True:
                            print("reset")
                            new_vector = torch.randn(new_vector.shape[1], device=self.ori_device, dtype=self.ori_dtype)
                            for existing_vec in existing:
                                projection = (new_vector @ existing_vec) / (existing_vec @ existing_vec)
                                new_vector = new_vector - projection * existing_vec
                            new_vector = F.normalize(new_vector, dim=0)
                            if torch.norm(new_vector) > 1e-3:
                                break
                    current_tensor[i, :] = new_vector
                self.train_keys_vit[pool_name] = nn.Parameter(current_tensor, requires_grad=True)   
        for pool_name in self.train_keys_llm.keys():
            if self.train_keys_llm[pool_name] is not None:
                current_tensor = self.train_keys_llm[pool_name].data.clone()
                vector_num, vector_dim = current_tensor.shape
                for i in range(vector_num):
                    if i == 0:
                        new_vector = current_tensor[0, :]
                        new_vector = F.normalize(new_vector, dim=0)
                        current_tensor[0, :] = new_vector
                        continue
                    existing = current_tensor[0:i,:]
                    if i < vector_dim:
                        new_vector = current_tensor[i, :]
                        for existing_vec in existing:
                            projection = (new_vector @ existing_vec) / (existing_vec @ existing_vec)
                            new_vector = new_vector - projection * existing_vec
                        new_vector = F.normalize(new_vector, dim=0)
                    else:
                        new_vector = current_tensor[i, :]
                        new_vector = F.normalize(new_vector, dim=0)
                    if torch.norm(new_vector) < 1e-3:
                        while True:
                            print("reset")
                            new_vector = torch.randn(new_vector.shape[1], device=self.ori_device, dtype=self.ori_dtype)
                            for existing_vec in existing:
                                projection = (new_vector @ existing_vec) / (existing_vec @ existing_vec)
                                new_vector = new_vector - projection * existing_vec
                            new_vector = F.normalize(new_vector, dim=0)
                            if torch.norm(new_vector) > 1e-3:
                                break
                    current_tensor[i, :] = new_vector
                self.train_keys_llm[pool_name] = nn.Parameter(current_tensor, requires_grad=True) 


    def _load_pools(self):
        llm_kwargs = {
            "share_size": self.lrp_config.llm_share_rank_size, 
            "static_size": self.lrp_config.llm_static_rank_size,
            "train_size": self.lrp_config.llm_train_rank_size,
            "top_rank": self.lrp_config.llm_top_rank,
            "freeze_share": self.lrp_config.llm_freeze_share,
            "ori_device": self.ori_device, 
            "ori_dtype": self.ori_dtype, 
            "get_score": self.get_score,
            "lora_alpha": self.lrp_config.llm_lora_alpha,
            "lora_dropout": self.lrp_config.llm_lora_dropout,
        }
        vit_kwargs = {
            "static_size": self.lrp_config.vit_static_prompt_size, 
            "train_size": self.lrp_config.vit_train_prompt_size, 
            "top_rank": self.lrp_config.vit_top_rank,
            "len": self.lrp_config.vit_prompt_len, 
            "dim": self.lrp_config.vit_hidden_size, 
            "ori_device": self.ori_device, 
            "ori_dtype": self.ori_dtype,
            "get_score": self.get_score,
            "vit_prompt_alpha": self.lrp_config.vit_prompt_alpha,
        }
        mm_kwargs = {
            "static_size": self.lrp_config.mm_static_rank_size,
            "train_size": self.lrp_config.mm_train_rank_size,
            "top_rank": self.lrp_config.mm_top_rank,
            "ori_device": self.ori_device, 
            "ori_dtype": self.ori_dtype, 
            "get_score": self.get_score,
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
                new_module = LlmLrpLinear(pool_name, in_features, out_features, **llm_kwargs)                   
                self._replace_module(parent, target_name_, new_module, target)
        if self.lrp_config.vit_target_modules is not None:
            for pool_name in self.lrp_config.vit_target_modules.keys():
                self.create_key_pool("vit", "llm", pool_name)
                self.create_key_pool("vit", "vit", pool_name)
                for target_name in self.lrp_config.vit_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    normalized_shape = target.normalized_shape
                    new_module = VitLrpLayerNorm(pool_name, normalized_shape, **vit_kwargs)
                    self._replace_module(parent, target_name_, new_module, target)
        if self.lrp_config.mm_target_modules is not None:
            for pool_name in self.lrp_config.mm_target_modules.keys():
                self.create_key_pool("mm", "llm", pool_name)
                self.create_key_pool("mm", "vit", pool_name)
                for target_name in self.lrp_config.mm_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    bias = target.bias is not None
                    mm_kwargs['bias'] = bias
                    in_features, out_features = target.in_features, target.out_features
                    new_module = mmLrpLinear(pool_name, in_features, out_features, **mm_kwargs)                   
                    self._replace_module(parent, target_name_, new_module, target)

    def create_key_pool(self, module_type: str, key_type: str, pool_name: str):
        if module_type == 'llm':
            static_size = self.lrp_config.llm_static_rank_size
            train_size = self.lrp_config.llm_train_rank_size
        elif module_type == 'mm': 
            static_size = self.lrp_config.mm_static_rank_size
            train_size = self.lrp_config.mm_train_rank_size
        else:  # vit
            static_size = self.lrp_config.vit_static_prompt_size
            train_size = self.lrp_config.vit_train_prompt_size
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
        if train_size is not None and train_size > 0:
            train_key_pool = torch.randn(train_size, k_hidden_size,
                            device=self.ori_device, dtype=self.ori_dtype)
            train_key_pool = F.normalize(train_key_pool, p=2, dim=1)
            train_param = nn.Parameter(train_key_pool, requires_grad=True)
            if key_type == 'llm':
                self.train_keys_llm[pool_name] = train_param
            else:
                self.train_keys_vit[pool_name] = train_param
        else:
            if key_type == 'llm':
                self.train_keys_llm[pool_name] = None
            else:
                self.train_keys_vit[pool_name] = None

    def get_score(self, pool_name):
        if self.llm_query is None and self.vit_query is None:
            return None, None
        k = self.vit_hidden_size / self.llm_hidden_size
        if self.train_keys_llm[pool_name] is not None and self.static_keys_llm[pool_name] is not None:
            if self.lrp_config.mode == "train":
                attn_static_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.static_keys_llm[pool_name], p=2, dim=1))
                attn_static_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.static_keys_vit[pool_name], p=2, dim=1))
                static_score = attn_static_llm + k * attn_static_vit
                attn_train_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.train_keys_llm[pool_name], p=2, dim=1))
                attn_train_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.train_keys_vit[pool_name], p=2, dim=1))
                train_score = attn_train_llm + k * attn_train_vit
                return static_score, train_score
            else:
                attn_static_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.static_keys_llm[pool_name], p=2, dim=1))
                attn_static_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.static_keys_vit[pool_name], p=2, dim=1))
                static_score = attn_static_llm + k * attn_static_vit
                attn_train_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.train_keys_llm[pool_name], p=2, dim=1))
                attn_train_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.train_keys_vit[pool_name], p=2, dim=1))
                train_score = attn_train_llm + k * attn_train_vit
                if self.generate_part == "past":
                    return static_score, None
                else:
                    return static_score, train_score
        elif self.train_keys_llm[pool_name] is not None:
            attn_train_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.train_keys_llm[pool_name], p=2, dim=1))
            attn_train_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.train_keys_vit[pool_name], p=2, dim=1))
            train_score = attn_train_llm + k * attn_train_vit
            return None, train_score
        else:
            attn_static_llm = torch.einsum('bd,kd->bk', self.llm_query, F.normalize(self.static_keys_llm[pool_name], p=2, dim=1))
            attn_static_vit = torch.einsum('bd,kd->bk', self.vit_query, F.normalize(self.static_keys_vit[pool_name], p=2, dim=1))
            static_score = attn_static_llm + k * attn_static_vit
            return static_score, None

    
    
    def get_model(self):
            return self.base_model.get_model()
    
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        has_image: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:   
        self.vit_query = None
        self.llm_query = None
        self.no_image = torch.tensor([i for i, item in enumerate(has_image) if item == 0]).to(dtype=torch.int)
        instruction_embdings = self.get_model().embed_tokens(instruction_ids)
        llm_query = get_llm_query(instruction_embdings, instruction_mask)
        if images is not None:
            img_select_feature_backup = self.base_model.get_model().get_vision_tower().select_feature
            self.base_model.get_model().get_vision_tower().select_feature = 'cls_patch'
            vit_query = self.base_model.get_model().get_vision_tower()(images)[:,0]
            self.base_model.get_model().get_vision_tower().select_feature = img_select_feature_backup
            vit_query = F.normalize(vit_query, p=2, dim=-1)
        else:
            vit_query = torch.zeros(llm_query.shape[0], self.vit_hidden_size, dtype=llm_query.dtype, device=llm_query.device)
        vit_query = vit_query.to(dtype=llm_query.dtype, device=llm_query.device)
        vit_query[self.no_image,:] = 0
        self.llm_query = llm_query.detach()
        self.vit_query = vit_query.detach()
        return self.base_model.forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            images = images,
            image_sizes = image_sizes,
            return_dict = return_dict,
            **kwargs
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        instruction_ids: Optional[torch.FloatTensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        self.llm_query = None
        self.vit_query = None
        instruction_embdings = self.get_model().embed_tokens(instruction_ids)
        llm_query = get_llm_query(instruction_embdings, None)
        if images is not None:
            img_select_feature_backup = self.base_model.get_model().get_vision_tower().select_feature
            self.base_model.get_model().get_vision_tower().select_feature = 'cls_patch'
            vit_query = self.base_model.get_model().get_vision_tower()(images)[:,0]      
            self.base_model.get_model().get_vision_tower().select_feature = img_select_feature_backup
            vit_query = F.normalize(vit_query, p=2, dim=-1)
        else:
            vit_query = torch.zeros(llm_query.shape[0], self.vit_hidden_size, dtype=llm_query.dtype, device=llm_query.device)
        self.llm_query = llm_query.detach()
        self.vit_query = vit_query.detach()
        if self.task_llm_query_pool is not None and self.task_vit_query_pool is not None:
            k = self.vit_hidden_size / self.llm_hidden_size
            k = 1
            score = torch.sum(torch.einsum('kd,cd->kc', self.llm_query, self.task_llm_query_pool) + k * torch.einsum('kd,cd->kc', self.vit_query, self.task_vit_query_pool), dim=0)
            _, max_index = torch.max(score, dim=0)
            max_index = max_index.item()
            # print(max_index)
            if max_index == score.shape[0] - 1:
                self.generate_part = "new"
            else:
                self.generate_part = "past"
                
        return self.base_model.generate(
            inputs = inputs,
            images = images,
            image_sizes = image_sizes,
            **kwargs,
        )
    
    
    def get_q_loss(self):
        llm_query = self.llm_query
        vit_query = self.vit_query
        loss = []
        for pool_name in self.static_keys_llm.keys():
            train_key_vit = self.train_keys_vit[pool_name]
            train_key_vit_normalize = F.normalize(train_key_vit, p=2, dim=1)
            train_score_vit = torch.einsum('bd,kd->bk', vit_query, train_key_vit_normalize)
            train_key_llm = self.train_keys_llm[pool_name]
            train_key_llm_normalize = F.normalize(train_key_llm, p=2, dim=1)
            train_score_llm = torch.einsum('bd,kd->bk', llm_query, train_key_llm_normalize)
            vit_loss_mat = 1 - train_score_vit
            vit_loss_mat[self.no_image,:] = 0
            loss.append(torch.mean(vit_loss_mat) + torch.mean(1 - train_score_llm))
        return torch.mean(torch.stack(loss, dim=0))
    
    def get_k_loss(self):
        if self.lrp_config.vit_target_modules is not None:
            train_key_llm = self.train_keys_llm["vit"]
            train_key_vit = self.train_keys_vit["vit"]
            static_key_llm = self.static_keys_llm["vit"]
            static_key_vit = self.static_keys_vit["vit"]
            loss = torch.tensor(0.0, dtype=torch.float, device=self.ori_device)
            if train_key_llm is not None:
                BBT = torch.mm(train_key_llm, train_key_llm.t())
                I = torch.eye(BBT.shape[0], device=BBT.device)
                loss += torch.norm(BBT - I, p='fro')
                BBT = torch.mm(train_key_vit, train_key_vit.t())
                I = torch.eye(BBT.shape[0], device=BBT.device)
                loss += torch.norm(BBT - I, p='fro')
            if train_key_llm is not None and static_key_llm is not None:
                BBT_cross = torch.mm(train_key_llm, static_key_llm.t())
                loss += torch.norm(BBT_cross, p='fro')
                BBT_cross = torch.mm(train_key_vit, static_key_vit.t())
                loss += torch.norm(BBT_cross, p='fro')
            return loss
        else:
            0

    def get_v_loss(self):
        losses = []
        if self.lrp_config.vit_target_modules is not None:
            for pool_name in self.lrp_config.vit_target_modules.keys():
                for target_name in self.lrp_config.vit_target_modules[pool_name]:
                    parent, target, target_name_ = _get_submodules(self.base_model, target_name)
                    train_prompt_pool = target.train_prompt_pool
                    static_prompt_pool = target.static_prompt_pool
                    prompts_len = train_prompt_pool.shape[1] if train_prompt_pool is not None else static_prompt_pool.shape[1]
                    loss = torch.tensor(0.0, dtype=torch.float, device=self.ori_device)
                    for i in range(prompts_len):
                        loss = torch.tensor(0.0, dtype=torch.float, device=self.ori_device)
                        if train_prompt_pool is not None:
                            current_Train = train_prompt_pool[:, i, :]
                            BBT = torch.mm(current_Train, current_Train.t())
                            I = torch.eye(BBT.shape[0], device=BBT.device)
                            loss += torch.norm(BBT - I, p='fro')
                        if train_prompt_pool is not None and static_prompt_pool is not None:
                            current_Static = static_prompt_pool[:, i, :]
                            BBT_cross = torch.mm(current_Train, current_Static.t())
                            loss += torch.norm(BBT_cross, p='fro')
                    losses.append(loss)
            return torch.stack(losses).sum()
        else:
            return 0

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