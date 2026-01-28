import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    # 检查提供的视觉塔是否为绝对路径并且存在
    is_absolute_path_exists = os.path.exists(vision_tower) if vision_tower else False
    
    # 是否启用多尺度处理
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    # 判断是否应该加载本地文件、OpenAI、Laion 或 ShareGPT4V 模型
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            # 如果启用了多尺度处理，则返回 CLIPVisionTowerS2 实例
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            # 否则返回标准的 CLIPVisionTower 实例
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 如果视觉塔名称不符合任何已知模式，则抛出异常
    raise ValueError(f'Unknown vision tower: {vision_tower}')