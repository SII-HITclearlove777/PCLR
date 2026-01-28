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



# 导入必要的类型提示模块
from typing import Optional, Tuple

import torch

# 导入必要的transformers模块
from transformers import AutoConfig, AutoModelForCausalLM, \
                         MptConfig, MptForCausalLM, MptModel
from myllava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# 定义LlavaMpt的配置类，继承自MptConfig
class LlavaMptConfig(MptConfig):
    model_type = "llava_mpt"  # 设置模型类型为llava_mpt


# 定义LlavaMpt模型类，继承自LlavaMetaModel和MptModel
class LlavaMptModel(LlavaMetaModel, MptModel):
    config_class = LlavaMptConfig  # 指定配置类

    def __init__(self, config: MptConfig):
        # 将config.d_model赋值给hidden_size，确保模型维度一致性
        config.hidden_size = config.d_model
        super(LlavaMptModel, self).__init__(config)
    
    # 定义token嵌入方法
    def embed_tokens(self, x):
        return self.wte(x)  # 使用词表嵌入层处理输入


# 定义LlavaMpt因果语言模型类，继承自MptForCausalLM和LlavaMetaForCausalLM
class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMptConfig  # 指定配置类
    supports_gradient_checkpointing = True  # 支持梯度检查点功能

    def __init__(self, config):
        # 初始化父类
        super(MptForCausalLM, self).__init__(config)

        # 初始化transformer模型和语言模型头
        self.transformer = LlavaMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 进行模型的后初始化处理
        self.post_init()

    # 获取模型的方法
    def get_model(self):
        return self.transformer

    # 设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMptModel):
            module.gradient_checkpointing = value

    # 模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入ID
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 过去的key-value对
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        labels: Optional[torch.Tensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式
        images=None):  # 图像输入

        # 准备多模态输入和标签
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        # 调用父类的forward方法
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # 为生成任务准备输入的方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)  # 从kwargs中提取图像输入
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        _inputs['images'] = images  # 将图像添加到输入字典中
        return _inputs


# 注册模型配置和模型类到AutoConfig和AutoModelForCausalLM
AutoConfig.register("llava_mpt", LlavaMptConfig)
AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM)