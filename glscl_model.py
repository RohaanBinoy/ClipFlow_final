import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalInteractionModule(nn.Module):
    """Parameter-free module for coarse-grained sentence-video alignment."""
    def __init__(self, tau=5.0):
        super().__init__()
        self.tau = tau

    def forward(self, video_features, text_global):
        sim = torch.einsum('bd,bfd->bf', text_global, video_features) / self.tau
        alpha = F.softmax(sim, dim=-1)
        video_global_guided = torch.einsum('bf,bfd->bd', alpha, video_features)
        return video_global_guided

class LocalInteractionModule(nn.Module):
    """Shared learnable queries to capture latent semantic concepts."""
    def __init__(self, embed_dim=512, num_queries=8, num_heads=8, num_layers=3):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features):
        B = features.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1) 
        concept_features = self.transformer(tgt=q, memory=features)
        return concept_features

class GLSCLModel(nn.Module):
    """The master Global-Local Semantic Consistent Learning architecture."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.global_interaction = GlobalInteractionModule(tau=5.0)
        self.local_interaction = LocalInteractionModule(embed_dim=embed_dim, num_queries=8)
        self.xi = 0.5 

    def forward(self, video_frame_features, text_word_features, text_global_feature):
        video_global_guided = self.global_interaction(video_frame_features, text_global_feature)
        c_v = self.local_interaction(video_frame_features)
        c_t = self.local_interaction(text_word_features)
        return video_global_guided, c_v