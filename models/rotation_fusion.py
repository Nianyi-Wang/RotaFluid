import torch
import torch.nn as nn


class HybridRotationFusion(nn.Module):
    
    def __init__(self, fluid_channels=4, rotation_channels=5, hidden_dim=32):
        super().__init__()
        
        self.fluid_channels = fluid_channels
        self.rotation_channels = rotation_channels
        
        self.physics_gate = nn.Sequential(
            nn.Linear(fluid_channels + rotation_channels + 3, 32),
            nn.ReLU(),
            nn.Linear(32, rotation_channels),
            nn.Sigmoid()
        )
        
        self.fluid_proj = nn.Linear(fluid_channels, hidden_dim)
        self.rotation_proj = nn.Linear(rotation_channels, hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Linear(hidden_dim * 2, fluid_channels + rotation_channels)
        
    def forward(self, pos, fluid_feats, rotation_feats, return_diagnostics=False):
        gate_input = torch.cat([pos, fluid_feats, rotation_feats], dim=-1)
        physics_gates = self.physics_gate(gate_input)
        
        modulated_rotation = physics_gates * rotation_feats
        
        fluid_emb = self.fluid_proj(fluid_feats)
        rotation_emb = self.rotation_proj(modulated_rotation)
        
        concat_emb = torch.cat([fluid_emb, rotation_emb], dim=-1)
        attention_weight = self.attention(concat_emb)
        
        weighted_rotation = attention_weight * rotation_emb
        weighted_fluid = (1 - attention_weight) * fluid_emb
        
        fused_emb = torch.cat([weighted_fluid, weighted_rotation], dim=-1)
        fused_feats = self.fusion(fused_emb)
        
        if return_diagnostics:
            diagnostics = {
                'physics_gates': physics_gates,
                'attention_weight': attention_weight,
                'effective_influence': physics_gates.mean(dim=-1, keepdim=True) * attention_weight,
                'modulated_rotation': modulated_rotation
            }
            return fused_feats, diagnostics
        
        return fused_feats
