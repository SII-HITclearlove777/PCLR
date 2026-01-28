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


# 导入所需的类型注解工具
from typing import List, Optional, Tuple, Union

# 导入PyTorch相关库
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 导入Transformers相关组件
from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

# 导入因果语言模型输出类型
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# 导入LLaVA基础架构组件
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    # 定义LLaVA-Mistral配置类，继承自MistralConfig
    model_type = "llava_mistral"  # 设定模型类型标识符


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    # LLaVA-Mistral模型类，继承自LlavaMetaModel和MistralModel
    config_class = LlavaMistralConfig  # 指定配置类

    def __init__(self, config: MistralConfig):
        # 初始化模型
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    # LLaVA-Mistral因果语言模型类，用于生成文本
    config_class = LlavaMistralConfig  # 指定配置类

    def __init__(self, config):
        # 初始化因果语言模型
        super(MistralForCausalLM, self).__init__(config)
        # 创建基础模型实例
        self.model = LlavaMistralModel(config)
        # 创建语言模型头部，用于词汇表预测
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 进行模型的后初始化处理
        self.post_init()

    def get_model(self):
        # 获取底层模型实例的方法
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,            # 输入词元ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,# 位置编码ID
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的key-value缓存
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        labels: Optional[torch.LongTensor] = None,      # 训练标签
        use_cache: Optional[bool] = None,               # 是否使用缓存
        output_attentions: Optional[bool] = None,       # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,    # 是否输出隐藏状态
        images: Optional[torch.FloatTensor] = None,     # 图像输入
        image_sizes: Optional[List[List[int]]] = None,  # 图像尺寸
        return_dict: Optional[bool] = None,             # 是否返回字典格式
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 前向传播函数

        if inputs_embeds is None:
            # 如果没有预计算的嵌入，则准备多模态输入
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # 调用父类的前向传播方法
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,         # 文本输入
        images: Optional[torch.Tensor] = None,         # 图像输入
        image_sizes: Optional[torch.Tensor] = None,    # 图像尺寸
        **kwargs,                                      # 其他参数
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 文本生成函数
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            # 如果提供了图像，处理多模态输入
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            # 如果没有图像，只处理文本输入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类的生成方法
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        # 为生成准备输入的辅助函数
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # 获取父类处理的输入
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 添加图像相关信息
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# 注册模型配置和模型类到AutoConfig和AutoModelForCausalLM
AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)