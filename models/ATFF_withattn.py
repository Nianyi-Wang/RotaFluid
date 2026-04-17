import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
# from torch_scatter import scatter_mean, scatter_max
import math
from models.ASCC import ContinuousConv as ASCC


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=3, mapping_size=256, scale=10):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = torch.matmul(x, self.B) * (2 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PhysicsInvariantAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        self.geom_mlp = nn.Sequential(
            nn.Linear(3, channels//2),
            nn.ReLU(),
            nn.Linear(channels//2, channels)
        )
        
        self.pos_proj = nn.Linear(32, 3)
    
    def forward(self, x, y, pos):
        """
        x, y: feature tensors [N, C]
        pos: position tensor [N, 3] or [N, 32]
        """
        if pos.shape[-1] == 32:
            pos = self.pos_proj(pos)
        
        batch_size = x.shape[0]
        
        pos_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(pos_diff, dim=-1)
        
        geom_features = self.geom_mlp(pos_diff)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        
        q = self.q_proj(x).view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(y.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).view(y.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        batch_size, num_heads = q.size(0), q.size(1)
        dist = dist.unsqueeze(0).unsqueeze(1)
        dist = dist.expand(batch_size, num_heads, -1, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        geom_features_expanded = geom_features.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        geom_features_per_head = geom_features_expanded.view(
            x.size(0), pos_diff.size(0), pos_diff.size(1), self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        
        q_per_head = q.unsqueeze(-2)
        geom_attn = torch.matmul(q_per_head, geom_features_per_head.transpose(-2, -1)).squeeze(-2)
        
        attn = attn + geom_attn - dist
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.size(0), -1, self.num_heads * self.head_dim)
        
        if x.size(0) == 1:
            out = out.squeeze(0)
            
        return out

class SpatioTemporalMemoryd(nn.Module):
    def __init__(self, channels, memory_length=5):
        super().__init__()
        self.memory_length = memory_length
        self.memory = None
        
        self.update_gate = nn.Sequential(
            nn.Linear(channels*2, channels),
            nn.Sigmoid()
        )
        
        self.temporal_attn = nn.Sequential(
            nn.Linear(channels, channels//2),
            nn.Tanh(),
            nn.Linear(channels//2, 1)
        )
        
    def forward(self, current_state):
        batch_size = current_state.shape[0]
        
        if self.memory is None:
            self.memory = torch.zeros(self.memory_length, batch_size, current_state.shape[-1]).to(current_state.device)
        
        update_signal = self.update_gate(torch.cat([current_state, self.memory[0]], dim=-1))
        
        new_memory = torch.cat([current_state.unsqueeze(0), self.memory[:-1]], dim=0)
        self.memory = update_signal.unsqueeze(0) * new_memory + (1 - update_signal.unsqueeze(0)) * self.memory
        
        attn_weights = self.temporal_attn(self.memory).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1)
        
        temporal_context = torch.sum(self.memory * attn_weights, dim=0)
        return temporal_context


class SpatioTemporalMemory(nn.Module):
    def __init__(self, channels, memory_length=5):
        super().__init__()
        self.memory_length = memory_length
        self.channels = channels
        
        self.update_gate = nn.Sequential(
            nn.Linear(channels*2, channels),
            nn.Sigmoid()
        )
        
        self.temporal_attn = nn.Sequential(
            nn.Linear(channels, channels//2),
            nn.Tanh(),
            nn.Linear(channels//2, 1)
        )
        
        self.batch_memories = {}
        
    def forward(self, current_state):
        batch_size = current_state.shape[0]
        device = current_state.device
        
        if batch_size not in self.batch_memories:
            self.batch_memories[batch_size] = torch.zeros(
                self.memory_length, batch_size, self.channels, device=device
            )
        
        memory = self.batch_memories[batch_size]
        
        update_signal = self.update_gate(torch.cat([current_state, memory[0]], dim=-1))
        
        new_memory = torch.cat([current_state.unsqueeze(0), memory[:-1]], dim=0)
        memory = update_signal.unsqueeze(0) * new_memory + (1 - update_signal.unsqueeze(0)) * memory
        
        self.batch_memories[batch_size] = memory
        
        attn_weights = self.temporal_attn(memory).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1)
        
        temporal_context = torch.sum(memory * attn_weights, dim=0)
        return temporal_context
    
    def reset_memory(self):
        self.batch_memories = {}


class EnhancedPositionEncoder(nn.Module):
    def __init__(self, input_dim=3, mapping_size=32, scale=10.0, use_learned=True, num_scales=4):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.use_learned = use_learned
        self.num_scales = num_scales
        
        self.bandwidths = nn.Parameter(
            torch.tensor([scale / (2**i) for i in range(num_scales)], dtype=torch.float32),
            requires_grad=False
        )
        
        if use_learned:
            self.learned_encoding = nn.Sequential(
                nn.Linear(input_dim, mapping_size),
                nn.LayerNorm(mapping_size),
                nn.GELU(),
                nn.Linear(mapping_size, mapping_size),
                nn.LayerNorm(mapping_size),
                nn.GELU()
            )
        
        total_dim = mapping_size * (num_scales*2 + (1 if use_learned else 0))
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, mapping_size),
            nn.LayerNorm(mapping_size),
            nn.GELU()
        )


        
    def fourier_mapping(self, pos, bandwidth):
        B = torch.randn(self.input_dim, self.mapping_size, device=pos.device) * bandwidth
        
        proj = torch.matmul(pos, B)
        
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
    def forward(self, pos):
        """
        pos: position tensor [N, 3]
        """
        fourier_features = []
        for i in range(self.num_scales):
            feat = self.fourier_mapping(pos, self.bandwidths[i])
            fourier_features.append(feat)
        
        all_features = fourier_features
        
        if self.use_learned:
            learned_feat = self.learned_encoding(pos)
            all_features.append(learned_feat)
        
        concat_features = torch.cat(all_features, dim=-1)
        encoded_pos = self.fusion(concat_features)
        
        return encoded_pos


class PhysicsNeuralInteractionField(nn.Module):
    def __init__(self, channels=32, inter_channels=64, conv_type='cconv', memory_length=5):
        super().__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        
        self.pos_encoder = EnhancedPositionEncoder(input_dim=3, mapping_size=channels)
        
        self.attention = PhysicsInvariantAttention(channels)
        
        
        def Conv(name, in_channels, out_channels, activation=None):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ASCC
                
            conv = conv_fn(
                kernel_size=[4, 4, 4],
                activation=activation,
                align_corners=True,
                normalize=False,
                radius_search_ignore_query_points=True,
                in_channels=in_channels,
                filters=out_channels
            )
            return conv
        
        self.cconv1 = Conv("conv1", channels*3, inter_channels, None)
        self.layer_norm1 = nn.LayerNorm(inter_channels)
        self.activation1 = nn.GELU()
        
        self.cconv2 = Conv("conv2", inter_channels, channels, None)
        self.layer_norm2 = nn.LayerNorm(channels)
        
        self.physics_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        
        self.gradient_reg = nn.Sequential(
            nn.Linear(channels, channels//2),
            nn.ReLU(),
            nn.Linear(channels//2, 1),
            nn.Sigmoid()
        )
        
        self.pos_fusion = nn.Sequential(
            nn.Linear(channels*2, channels),
            nn.GELU()
        )
        
    def forward(self, x, y, pos, velocity=None, timestep=None):
        pos_embed = self.pos_encoder(pos)
        
        x_with_pos = self.pos_fusion(torch.cat([x, pos_embed], dim=-1))
        y_with_pos = self.pos_fusion(torch.cat([y, pos_embed], dim=-1))
        
        attn_output = self.attention(x_with_pos, y_with_pos, pos)
        
        concat_features = torch.cat([x_with_pos,  y_with_pos,attn_output], dim=-1)
        
        fused_features = self.cconv1(concat_features, pos, pos, self.filter_extent)
        fused_features = self.layer_norm1(fused_features)
        fused_features = self.activation1(fused_features)
        
        output_features = self.cconv2(fused_features, pos, pos, self.filter_extent)
        output_features = self.layer_norm2(output_features)
        
        if velocity is not None and timestep is not None:
            grad_constraint = self.gradient_reg(velocity)
            output_features = output_features * grad_constraint
        
        gate = self.physics_gate(output_features)
        final_output = x * gate + y * (1 - gate)
        return final_output
    
    def reset_memory(self):
        pass


