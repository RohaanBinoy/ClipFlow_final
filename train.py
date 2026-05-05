import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import open_clip
import cv2
from PIL import Image
import numpy as np
import os

# Import your custom engine
from clipflow_engine import CLIPflow_Engine

# ==========================================
# 1. THE GLSCL EXACT LOSS FUNCTIONS
# ==========================================

def calculate_infonce_loss(scores, logit_scale):
    """Standard Contrastive Loss (Used for the Global Highlighter)"""
    scaled_scores = scores * logit_scale.exp()
    labels = torch.arange(scores.shape[0], device=scores.device)
    loss_v2t = F.cross_entropy(scaled_scores, labels)
    loss_t2v = F.cross_entropy(scaled_scores.t(), labels)
    return (loss_v2t + loss_t2v) / 2

def calculate_intra_diversity_loss(concepts):
    """
    IDL: Forces the 8 local detectives to look at DIFFERENT things.
    Penalizes them if their cosine similarity to each other is too high.
    """
    # concepts shape: [Batch, 8, 512]
    concepts_norm = F.normalize(concepts, dim=-1)
    
    # Calculate similarity of the 8 concepts against each other
    sim_matrix = torch.bmm(concepts_norm, concepts_norm.transpose(1, 2))
    
    # We ignore the diagonal (a detective is obviously 100% similar to itself)
    mask = torch.eye(sim_matrix.size(1), device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(mask, 0.0)
    
    # Calculate the penalty (MSE of the off-diagonal similarities)
    idl_loss = (sim_matrix ** 2).mean()
    return idl_loss

def calculate_master_loss(global_scores, c_v, c_t, logit_scale):
    """
    The exact Total Loss function defined in the GLSCL paper.
    """
    # 1. Standard InfoNCE on the Global Scores
    loss_global = calculate_infonce_loss(global_scores, logit_scale)
    
    # 2. Inter-Consistency Loss (ICL) on the Local Concepts
    # We measure InfoNCE across the averaged concepts
    c_v_mean = c_v.mean(dim=1)
    c_t_mean = c_t.mean(dim=1)
    local_scores = torch.matmul(F.normalize(c_v_mean, dim=-1), F.normalize(c_t_mean, dim=-1).t())
    loss_icl = calculate_infonce_loss(local_scores, logit_scale)
    
    # 3. Intra-Diversity Loss (IDL) for both Video and Text concepts
    loss_idl_v = calculate_intra_diversity_loss(c_v)
    loss_idl_t = calculate_intra_diversity_loss(c_t)
    loss_idl = (loss_idl_v + loss_idl_t) / 2
    
    # Final combined formula from the paper!
    total_loss = loss_global + loss_icl + loss_idl
    return total_loss


# ==========================================
# 2. THE FEATURE EXTRACTOR
# ==========================================
# We reuse the hack logic here so we can generate the true matrices during training
def extract_training_features(video_path, text_prompt, clip_model, preprocess, tokenizer, device):
    # --- TEXT ---
    text_tokens = tokenizer([text_prompt]).to(device)
    x_t = clip_model.token_embedding(text_tokens).to(clip_model.dtype)
    x_t = x_t + clip_model.positional_embedding.to(clip_model.dtype)
    x_t = x_t.permute(1, 0, 2)
    x_t = clip_model.transformer(x_t)
    x_t = x_t.permute(1, 0, 2)
    x_t = clip_model.ln_final(x_t)
    
    t_global = x_t[torch.arange(x_t.shape[0]), text_tokens.argmax(dim=-1)]
    if clip_model.text_projection is not None:
        t_global = t_global @ clip_model.text_projection
        t_words = x_t @ clip_model.text_projection
    else:
        t_words = x_t

    # --- VIDEO ---
    cap = cv2.VideoCapture(video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total_f - 1), 5, dtype=int)
    frames = [preprocess(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))) 
              for idx in idxs if cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)) and (fr := cap.read()[1]) is not None]
    cap.release()
    
    if len(frames) == 0:
        return None
        
    v_tensor = torch.stack(frames).to(device)
    visual = clip_model.visual
    x_v = visual.conv1(v_tensor)
    x_v = x_v.reshape(x_v.shape[0], x_v.shape[1], -1).permute(0, 2, 1)
    
    cls_token = visual.class_embedding.to(x_v.dtype) + torch.zeros(x_v.shape[0], 1, x_v.shape[-1], dtype=x_v.dtype, device=device)
    x_v = torch.cat([cls_token, x_v], dim=1)
    x_v = x_v + visual.positional_embedding.to(x_v.dtype)
    x_v = visual.ln_pre(x_v)
    x_v = x_v.permute(1, 0, 2)
    x_v = visual.transformer(x_v)
    x_v = x_v.permute(1, 0, 2)
    
    if visual.proj is not None:
        x_v = x_v @ visual.proj
        
    v_global = x_v[:, 0, :].unsqueeze(0)
    v_patches = x_v[:, 1:, :].unsqueeze(0)
    
    return v_global.float(), v_patches.float(), t_global.float(), t_words.float()

# ==========================================
# 3. THE MASTER TRAINING LOOP
# ==========================================
def train_clipflow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Starting CLIPflow Training on {device.upper()}")
    
    # 1. Load frozen CLIP (We DO NOT train these weights!)
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.to(device).eval()
    
    # 2. Load your custom engine (We DO train these weights!)
    engine = CLIPflow_Engine().to(device)
    engine.train() # Set to training mode!
    
    # 3. Set up the Optimizer (AdamW)
    # We explicitly tell it to only update the engine's parameters, NOT the CLIP backbone
    optimizer = optim.AdamW(engine.parameters(), lr=1e-4, weight_decay=0.01)
    
    # --- DUMMY DATASET FOR TESTING THE PIPELINE ---
    # In reality, this would be a loop reading thousands of videos from a folder
    dataset = [
        {"video": "car_video.mp4", "prompt": "a red car driving"},
        {"video": "dog_video.mp4", "prompt": "a dog running in the park"}
    ]
    
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        
        # Zero the gradients before the batch
        optimizer.zero_grad()
        
        scores_matrix = [] # We need to build a Batch x Batch score matrix
        
        # Step A: Forward Pass (Get all the scores)
        # Note: A real implementation batches these parallelly for speed
        for item in dataset:
            if not os.path.exists(item["video"]):
                print(f"Missing {item['video']}, skipping...")
                continue
                
            with torch.no_grad(): # Don't track gradients for the feature extraction
                features = extract_training_features(item["video"], item["prompt"], clip_model, preprocess, tokenizer, device)
            
            if features is None: continue
            v_global, v_patches, t_global, t_words = features
            
            # Now run the engine WITH gradient tracking!
            final_score, _, _ = engine(v_global, v_patches, t_global, t_words)
            scores_matrix.append(final_score)
            
        if not scores_matrix:
            print("No valid videos found. Please add videos to the folder!")
            return
            
        # Convert list of 1x1 tensors to a proper matrix (for this dummy loop, we just use a 1D tensor and calculate a simplified loss)
        # In a real batch of N, this would be an NxN matrix.
        scores_tensor = torch.cat(scores_matrix).view(len(scores_matrix), 1) 
        
        # Step B: Calculate the Loss (How wrong were the detectives?)
        # Since our dummy dataset processes 1 by 1 instead of a full NxN grid, we do a simplified MSE loss against a target of 1.0
        # For a full dataset, you'd use the calculate_infonce_loss function above!
        target = torch.ones_like(scores_tensor)
        loss = F.mse_loss(scores_tensor, target)
        
        # Step C: Backpropagation (Calculus magic!)
        loss.backward()
        
        # Step D: Update the Weights
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
    # 4. Save the trained brain!
    torch.save(engine.state_dict(), "clipflow_trained_weights.pt")
    print("✅ Training Complete! Weights saved as 'clipflow_trained_weights.pt'")

if __name__ == "__main__":
    train_clipflow()