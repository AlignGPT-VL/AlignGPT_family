import os
from ..multimodal_components.clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from ..multimodal_components.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        if use_s2:
            # print('here S2')
            if "siglip" in vision_tower.lower():
                print("here siglip s2")
                return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                print("here clip s2")
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            if "siglip" in vision_tower.lower():
                print("here siglip")
                return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                print("here clip")
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
