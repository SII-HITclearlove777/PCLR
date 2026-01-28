import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False  # 标记模型是否已加载

        # 初始化参数
        self.vision_tower_name = vision_tower  # 视觉塔的名称或路径
        self.select_layer = args.mm_vision_select_layer  # 选择哪一层的隐藏状态作为输出
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')  # 选择的特征类型

        # 如果不延迟加载或需要解冻视觉塔，则立即加载模型
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # 否则只加载配置
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        # 加载图像处理器和视觉塔模型，并冻结模型参数
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True  # 标记模型已加载

    def feature_select(self, image_forward_outs):
        # 从指定层获取特征
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            # 如果选择的是 patch 特征，则去掉 [CLS] token
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # 如果选择的是包含 [CLS] token 的 patch 特征，则保留所有
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if isinstance(images, list):  # 如果输入是图像列表，则逐个处理
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:  # 如果输入是单个张量，则直接处理
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        # 返回一个虚拟特征，通常用于初始化或其他用途
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # 返回视觉塔模型的数据类型
        return self.vision_tower.dtype

    @property
    def device(self):
        # 返回视觉塔模型所在的设备（CPU 或 GPU）
        return self.vision_tower.device

    @property
    def config(self):
        # 返回模型配置，如果未加载则返回 cfg_only
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        # 返回模型的隐藏层大小
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        # 返回每边的补丁数量
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        # 返回总的补丁数量
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        # 多尺度设置
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running:\npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # 修改预处理尺寸为最大尺度
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        """ 加载 CLIP 视觉模型，并调整预处理尺寸 """
        super().load_model(device_map)

        # 设置预处理尺寸为最大的尺度
        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def forward_feature(self, images):
        """ 提取特征，不进行多尺度处理 """
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        """ 使用多尺度前向传播函数处理图像 """
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        """ 返回多尺度情况下的隐藏层大小 """
        return self.config.hidden_size * len(self.s2_scales)
