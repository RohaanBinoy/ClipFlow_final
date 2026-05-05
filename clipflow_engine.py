import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIPflow_Engine(nn.Module):
    """The Corrected Phase I Architecture trained on Kaggle"""
    def __init__(self, embed_dim=512, num_queries=8):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.concept_proj = nn.Linear(embed_dim, num_queries)
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.xi = 0.5

    def global_interaction(self, v_g, t_g):
        t_g_expanded = t_g.unsqueeze(1) 
        v_g_guided, _ = self.global_attn(t_g_expanded, v_g, v_g)
        return v_g_guided.squeeze(1)

    def local_interaction(self, local_features):
        concept_weights = F.softmax(self.concept_proj(local_features), dim=-1)
        concepts = torch.bmm(concept_weights.transpose(1, 2), local_features)
        return concepts

    def forward(self, video_global, video_patches, text_global, text_words):
        # 1. Global Phase 
        v_g_guided = self.global_interaction(video_global, text_global)
        
        # 2. Local Phase
        B, num_frames, P, D = video_patches.shape
        video_patches_flat = video_patches.reshape(B, num_frames * P, D)
        
        c_v = self.local_interaction(video_patches_flat)
        c_t = self.local_interaction(text_words)
        
        # 3. Score Calculation 
        v_g_norm = F.normalize(v_g_guided, dim=-1)
        t_g_norm = F.normalize(text_global, dim=-1)
        score_global = torch.sum(v_g_norm * t_g_norm, dim=-1)
        
        # 🎯 THE FIX: Many-to-Many Concept Alignment Matrix 
        c_v_norm = F.normalize(c_v, dim=-1)
        c_t_norm = F.normalize(c_t, dim=-1)
        local_sim_matrix = torch.einsum('bmd,bnd->bmn', c_v_norm, c_t_norm)
        score_local = local_sim_matrix.max(dim=-1)[0].mean(dim=-1)
        
        # 4. Streamlit UI Score Scaling
        raw_score = score_global + (self.xi * score_local)
        
        # Map the raw cosine decimals to a beautiful 0% to 100% UI confidence curve
        ui_score = (raw_score + 0.1) * 2.0 
        ui_score = torch.clamp(ui_score, min=0.0, max=1.0)
        
        return ui_score, score_global, score_local