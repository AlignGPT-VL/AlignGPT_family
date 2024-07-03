import torch
import torch.nn as nn
from src.utils.constants import N_IMAGE_TOKEN


def build_align(config):
    return nn.Parameter(torch.ones(config.n_indicators, 
                            config.hidden_size))
        

class AlignIndicator(nn.Module):
    def __init__(self, config, n_indicators=8) -> None:
        super().__init__()
        self.indicator_embs = nn.Parameter(nn.init.xavier_uniform_(
                torch.empty(config.n_indicators, 
                            config.hidden_size))
        )
    
    def __call__(self, ids) -> torch.Any:
        return self.indicator_embs[ids]
    
    def lin_comb(self, b_weight_scores):
        indicators = self.indicator_embs[:-1]
        final_indicators = torch.matmul(b_weight_scores, indicators) + self.indicator_embs[-1]

        return final_indicators
    
    def lin_comb_local(self, b_weight_scores):
        indicators = self.indicator_embs[:-1]
        final_indicators = torch.cat([torch.matmul(b_weight_scores, indicators), self.indicator_embs[-1].expand(b_weight_scores.shape[0], -1)[:, 0:0]], dim=1)

        return final_indicators
    
    def mean_align(self):
        indicators = self.indicator_embs[:-1]
        final_indicators = torch.mean(indicators, dim=0, keepdim=True) + self.indicator_embs[-1]

        return final_indicators
    
    def global_align(self):
        indicators = self.indicator_embs[:-1]
        final_indicators = torch.cat([self.indicator_embs[-1].unsqueeze(dim=0), torch.mean(indicators, dim=0, keepdim=True)[:, 0:0]], dim=1)

        return final_indicators
        
class GatedWeightLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.dim = config.hidden_size
        self.mlp = nn.Sequential(*[nn.Linear(N_IMAGE_TOKEN, 256), # TODO: 图片长度如果有变化
                                   nn.GELU(),
                                   nn.Linear(256, config.n_indicators-1)])

    def forward(self, image_embeds, text_embeds, text_attention_mask):
        scale = torch.sum(text_attention_mask.int(), dim=-1, keepdim=True).to(dtype=text_embeds.dtype)
        avg_text_embeds = torch.sum(text_embeds, dim=1) / (scale + 1e-8)
        # 16, 769, 4096
        # 16, 576, 4096
        # print(image_embeds.shape)
        dots = torch.sum(image_embeds * avg_text_embeds.unsqueeze(dim=1), dim=-1)  
        # 16, 768
        # 16, 576 
        # print(dots.shape)
        scores = self.mlp(dots) # [bs, 7]
        scores = scores.softmax(dim=-1)

        return scores
    
    def dummy_forward(self, bs, device, dtype):
        embs1 = torch.ones([bs, N_IMAGE_TOKEN], dtype=dtype, device=device)
        out = self.mlp(embs1)

        return out



