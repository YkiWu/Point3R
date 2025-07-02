import numpy as np  
import torch
import torch.nn as nn
import math

class RoPE3DContinuous(nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.base_list = [10.0, 100.0, 1000.0, 10000.0]
        self.F0 = F0

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def get_angle_emb(self, pos1d, D, device, dtype, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
        angle = self.F0 * pos1d.unsqueeze(-1) * inv_freq  # [..., ntokens, D//2]
        angle = angle.to(dtype)
        
        angle = torch.cat([angle, angle], dim=-1)  # [..., ntokens, D]
        angle = angle[..., :D]  
        return angle.cos(), angle.sin()

    def apply_rope1d_continuous(self, tokens, pos1d):
        all_encoded = []
        for base in self.base_list:
            cos, sin = self.get_angle_emb(pos1d, tokens.size(-1), tokens.device, tokens.dtype, base)
            cos = cos.unsqueeze(1)  # [B, 1, ntokens, D]
            sin = sin.unsqueeze(1)
            encoded = tokens * cos + self.rotate_half(tokens) * sin
            all_encoded.append(encoded)
        return torch.mean(torch.stack(all_encoded, dim=0), dim=0)
        
    def forward(self, tokens, positions):
        len_tokens = tokens.shape[2]
        len_positions = positions.shape[1]
        assert len_tokens == len_positions or len_tokens == len_positions + 1, f"tokens: {len_tokens}, positions: {len_positions}"
        if len_tokens != len_positions:
            pose_token = tokens[:, :, 0, :].unsqueeze(2)
            img_token = tokens[:, :, 1:, :]
        else:
            img_token = tokens
            
        D = img_token.size(-1) // 3 
        remain = img_token.size(-1) % 3
        if remain == 0:
            x_token, y_token, z_token = img_token.chunk(3, dim=-1)
        else:
            x_token = img_token[..., :D]
            y_token = img_token[..., D:2*D]
            z_token = img_token[..., 2*D:]
        x_pos, y_pos, z_pos = positions.unbind(-1)  

        x_encoded = self.apply_rope1d_continuous(x_token, x_pos)
        y_encoded = self.apply_rope1d_continuous(y_token, y_pos)
        z_encoded = self.apply_rope1d_continuous(z_token, z_pos)
        
        img_token = torch.cat([x_encoded, y_encoded, z_encoded], dim=-1)
        
        if len_tokens != len_positions:
            tokens_return = torch.cat([pose_token, img_token], dim=2)
        else:
            tokens_return = img_token
        
        return tokens_return
