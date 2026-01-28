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

# 导入必要的类型注解工具
from typing import List, Optional, Tuple, Union

# 导入PyTorch相关库
import torch
import torch.nn as nn

# 导入Transformers相关组件
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# 导入LLaVA架构相关组件
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    """
    LLaVA模型的配置类，继承自LlamaConfig
    扩展了标准的Llama配置，添加了LLaVA特定的配置选项
    用于存储模型的超参数和配置信息
    """
    # 定义模型类型标识符，用于模型的识别和加载
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """
    LLaVA的核心模型类
    - 继承自LlavaMetaModel：提供多模态处理能力
    - 继承自LlamaModel：提供基础的Llama模型功能
    组合了视觉和语言模型的特性
    """
    # 指定该模型使用的配置类
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        # 初始化父类，设置模型的基本配置
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    LLaVA的因果语言模型类
    - 继承自LlamaForCausalLM：提供生成式语言模型的功能
    - 继承自LlavaMetaForCausalLM：提供多模态处理能力
    实现了完整的多模态对话生成功能
    """
    # 指定该模型使用的配置类
    config_class = LlavaConfig

    def __init__(self, config):
        # 初始化基础的Llama因果语言模型
        super(LlamaForCausalLM, self).__init__(config)
        # 创建模型实例
        self.model = LlavaLlamaModel(config)
        # 保存预训练时的张量并行度配置
        self.pretraining_tp = config.pretraining_tp
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 创建语言模型的输出层，将隐藏状态映射到词汇表大小的输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 执行模型的后初始化处理（如权重初始化等）
        self.post_init()

    def get_model(self):
        """
        获取模型的内部表示
        返回：内部的LlavaLlamaModel实例
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,                    # 输入的token ID序列
        attention_mask: Optional[torch.Tensor] = None,         # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,       # 位置编码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的key/value缓存
        inputs_embeds: Optional[torch.FloatTensor] = None,     # 直接的嵌入输入
        labels: Optional[torch.LongTensor] = None,             # 训练标签
        use_cache: Optional[bool] = None,                      # 是否使用key/value缓存
        output_attentions: Optional[bool] = None,              # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,           # 是否输出隐藏状态
        images: Optional[torch.FloatTensor] = None,            # 图像输入
        image_sizes: Optional[List[List[int]]] = None,         # 图像尺寸信息
        return_dict: Optional[bool] = None,                    # 是否返回字典格式
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        模型的前向传播函数
        处理多模态输入（文本和图像），生成预测结果
        """
        # 如果没有直接提供词嵌入，则需要处理多模态输入
        if inputs_embeds is None:
            # 准备多模态输入（处理文本和图像）
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
        # 调用父类的前向传播方法处理准备好的输入
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
            return_dict=return_dict,
            **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,        # 文本输入
        images: Optional[torch.Tensor] = None,        # 图像输入
        image_sizes: Optional[torch.Tensor] = None,   # 图像尺寸
        **kwargs,                                     # 其他生成参数
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        生成函数，用于模型推理
        处理多模态输入并生成文本输出
        """
        # 从kwargs中获取位置编码和注意力掩码
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # 确保不使用直接的嵌入输入
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        # 如果提供了图像，处理多模态输入
        if images is not None:
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

    def parent_generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        为生成准备输入数据
        处理文本输入和可选的图像输入
        """
        # 从kwargs中提取图像相关信息
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # 获取基础的输入准备

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        # 如果有图像数据，将其添加到输入字典中
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
            
        return inputs


# 注册LLaVA的配置和模型类到AutoConfig和AutoModelForCausalLM中
# 这样可以通过from_pretrained方法加载模型
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)