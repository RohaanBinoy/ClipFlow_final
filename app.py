import streamlit as st
import os
import numpy as np
from moviepy import VideoFileClip
import chromadb
import uuid
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from clipflow_engine import CLIPflow_Engine
import open_clip

# --- 1. THE REPLACEABLE MODEL ARCHITECTURE ---
class BaseAIModel:
    def get_video_embedding(self, video_path):
        raise NotImplementedError

    def get_text_embedding(self, text):
        raise NotImplementedError

class DummyModel(BaseAIModel):
    def get_video_embedding(self, video_path):
        return np.random.rand(512).tolist()

    def get_text_embedding(self, text):
        return np.random.rand(512).tolist()

class RealCLIPModel(BaseAIModel):
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

    def get_video_embedding(self, video_path):
        clip = VideoFileClip(video_path)
        frames = []
        for t in range(int(clip.duration)):
            frame = clip.get_frame(t)
            frames.append(Image.fromarray(frame))
        clip.close()
        
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
        
        video_embedding = image_features.mean(dim=0)
        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        return video_embedding.tolist()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()

class CLIP4ClipModel(BaseAIModel):
    def __init__(self):
        self.model_id = "Searchium-ai/clip4clip-webvid150k"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

    def get_video_embedding(self, video_path):
        clip = VideoFileClip(video_path)
        frames = []
        for t in range(int(clip.duration)):
            frame = clip.get_frame(t)
            frames.append(Image.fromarray(frame))
        clip.close()
        
        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
        
        video_embedding = image_features.mean(dim=0)
        video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
        return video_embedding.tolist()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_features = self.model.text_projection(text_outputs.pooler_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()

class FinalGLSCLModel(BaseAIModel):
    """GLSCL/DiCoSA Final Model Wrapper"""
    def __init__(self):
        import sys
        import os
        from types import SimpleNamespace
        from transformers import CLIPProcessor
        
        sys.path.append(os.path.join(os.path.dirname(__file__), 'final model'))
        from final_model import modeling
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.config = SimpleNamespace(
            interaction='no',
            agg_module='meanP',
            base_encoder='ViT-B/32',
            center=1,
            query_number=8,
            cross_att_layer=2,
            num_hidden_layers=4,
            temp=0.07,
            loss2_weight=1.0,
            alpha=1.0,
            beta=1.0,
            query_share=True,
            cross_att_share=True
        )
        
        self.model = modeling.DiCoSA(self.config).to(self.device).eval()
        
        self.weights_path = os.path.join("final model", "final_model.pt")
        if os.path.exists(self.weights_path):
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)
            
        # We'll use HuggingFace's processor to format images/text into the tensors DiCoSA expects
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_video_embedding(self, video_path):
        import cv2
        import torch
        from PIL import Image
        
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = getattr(self.config, 'num_frames', 8)
        idxs = np.linspace(0, max(0, total_f - 1), num_frames, dtype=int)
        
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            return np.zeros(512).tolist()
            
        imgs = [Image.fromarray(f) for f in frames]
        
        # 1. Preprocess the frames using CLIP's standard transforms
        inputs = self.processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device) # Shape: [num_frames, 3, 224, 224]
        
        # 2. DiCoSA expects an extra batch dimension: [batch_size, num_frames, channels, height, width]
        video_tensor = pixel_values.unsqueeze(0) 
        
        # 3. Create a mask (1s meaning all frames are valid)
        video_mask = torch.ones((1, len(frames))).to(self.device)
        
        with torch.no_grad():
            # Get the sequence of frame features natively via DiCoSA
            video_emb = self.model.get_video_feat(video_tensor, video_mask)
            # Average the frames
            video_emb = video_emb.mean(dim=1).squeeze()
            
            # ✨ THE FIX: Normalize the vector so ChromaDB math works!
            video_emb = F.normalize(video_emb, p=2, dim=-1)
            
            
        return video_emb.tolist()

    def get_text_embedding(self, text):
        import torch
        
        # Tokenize to get standard input_ids and attention_mask
        inputs = self.processor(
            text=[text], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        )
        text_ids = inputs["input_ids"].to(self.device)
        text_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # DiCoSA's get_text_feat returns (text_feat, cls_feat)
            text_feat, cls_feat = self.model.get_text_feat(text_ids, text_mask)
            # We want the global CLS token feature
            text_emb = cls_feat.squeeze()
            
            # ✨ THE FIX: Normalize the vector so ChromaDB math works!
            text_emb = F.normalize(text_emb, p=2, dim=-1)
            
        return text_emb.tolist()

class CLIPflowCustomModel(BaseAIModel):
    """The Uncompromised CLIPflow Phase I Engine"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load the frozen CLIP eyes
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model.to(self.device).eval()
        
        # 2. Load your pure, branded engine!
        self.engine = CLIPflow_Engine(embed_dim=512, num_queries=8).to(self.device).eval()
        
        # 🚨 THE MISSING MEMORIES: Load your Kaggle weights! 🚨
        self.weights_path = "clipflow_epoch2_batch4000.pt"
        
        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.engine.load_state_dict(checkpoint, strict=False)
            print(f"✅ SUCCESSFULLY LOADED AI BRAIN: {self.weights_path}")
        else:
            print(f"❌ ERROR: Could not find {self.weights_path}. The AI is guessing randomly!")
        
        # 3. Live memory cache (Bypassing ChromaDB for the complex math)
        self.video_cache = {}

    def extract_true_video_features(self, video_frames):
        """Hacks CLIP to get [CLS] tokens and spatial patches."""
        visual = self.clip_model.visual
        x = visual.conv1(video_frames)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        if visual.proj is not None:
            x = x @ visual.proj
            
        return x[:, 0, :], x[:, 1:, :] # Global, Patches

    def get_video_embedding(self, video_path):
        """Processes the video when the user clicks 'Process Video'"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        idxs = np.linspace(0, max(0, total_f - 1), 5, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        
        imgs = [self.preprocess(Image.fromarray(f)) for f in frames]
        if not imgs: return np.zeros(512).tolist()
             
        x = torch.stack(imgs).to(self.device)
        
        with torch.no_grad():
            video_global, video_patches = self.extract_true_video_features(x)
            
        # Store BOTH matrices in memory for the Search phase
        self.video_cache[video_path] = {
            "global": video_global.float(),
            "patches": video_patches.float()
        }
        return video_global.mean(dim=0).tolist() # Dummy return for Streamlit

    def get_text_embedding(self, text):
        return np.zeros(512).tolist()
    
# --- 2. THE DATABASE SETUP ---
client = chromadb.Client()
collection = client.get_or_create_collection("clipflow_db")

# --- 3. VIDEO PROCESSING ENGINE ---
def chop_video(video_path, chunk_length=5):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    chunks = []
    if not os.path.exists("chunks"):
        os.makedirs("chunks")

    for i in range(0, int(duration), chunk_length):
        start = i
        end = min(i + chunk_length, duration)
        chunk_path = f"chunks/clip_{start}_{end}.mp4"
        subclip = clip.subclipped(start, end)
        subclip.write_videofile(chunk_path, codec="libx264", audio_codec="aac", logger=None)
        chunks.append(chunk_path)
    return chunks

def extract_true_text_features(clip_model, text_tokens):
    """
    Hacks into the CLIP transformer to extract both the global sentence 
    embedding AND the individual word-level embeddings.
    """
    # ✨ FIX #1: Getting the correct model data type ✨
    model_dtype = clip_model.visual.conv1.weight.dtype
    
    # 1. Convert the text tokens into base embeddings
    x = clip_model.token_embedding(text_tokens).to(model_dtype)
    
    # 2. Add positional embeddings (so the AI knows word order)
    x = x + clip_model.positional_embedding.to(model_dtype)
    
    # 3. Pass it through the core Transformer brain
    x = x.permute(1, 0, 2)  # NLD -> LND (required by PyTorch transformers)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    
    # 4. Apply the final Layer Normalization
    x = clip_model.ln_final(x) 
    
    # 5. Extract the Global Feature (The [EOS] token summary)
    text_global = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
    
    # 6. Apply the final text projection bridge (if the model has one)
    if clip_model.text_projection is not None:
        text_global = text_global @ clip_model.text_projection
        x = x @ clip_model.text_projection # Project the individual words too!
        
    # Return BOTH the Forest (Global) and the Trees (Words)
    return text_global, x

# --- 4. FRONT END (STREAMLIT) ---
st.title("🌊 ClipFlow: AI Video Search")
st.write("Upload a video, and we will chop it up, embed it, and make it searchable!")

st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Choose AI Engine", 
    [
        "Original CLIP (Baseline)", 
        "Searchium CLIP4Clip (High Accuracy)", 
        "Clipper (Custom Local Model)",  
        "CLIPflow (Custom Engine)", 
        "GLSCL/DiCoSA (Final Model)",
        "Dummy Model (Fast Testing)"
    ]
)

confidence_threshold = st.sidebar.slider("Minimum Confidence Score (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

@st.cache_resource
def load_clipper(): return ClipperCustomModel()
@st.cache_resource
def load_baseline(): return RealCLIPModel()
@st.cache_resource
def load_clip4clip(): return CLIP4ClipModel()
@st.cache_resource
def load_clipflow(): return CLIPflowCustomModel()
@st.cache_resource
def load_final_model(): return FinalGLSCLModel()

if selected_model == "Original CLIP (Baseline)": model = load_baseline()
elif selected_model == "Searchium CLIP4Clip (High Accuracy)": model = load_clip4clip()
elif selected_model == "Clipper (Custom Local Model)": model = load_clipper()
elif selected_model == "CLIPflow (Custom Engine)": model = load_clipflow()
elif selected_model == "GLSCL/DiCoSA (Final Model)": model = load_final_model()
else: model = DummyModel()

uploaded_file = st.file_uploader("Upload a Video (mp4)", type=["mp4"])

if uploaded_file is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(temp_video_path)
    
    if st.button("Process Video"):
        try: client.delete_collection("clipflow_db")
        except: pass
        collection = client.create_collection("clipflow_db")
        
        if selected_model == "CLIPflow (Custom Engine)":
            model.video_cache.clear() # Clear the live memory

        with st.spinner("Chopping video into 5-second clips..."):
            chunk_paths = chop_video(temp_video_path)
            st.success(f"Successfully created {len(chunk_paths)} short clips!")
            
        with st.spinner("Running AI Model (This will take a moment...)"):
            for chunk in chunk_paths:
                embedding = model.get_video_embedding(chunk)
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk], 
                    ids=[str(uuid.uuid4())]
                )
            st.success("All clips embedded and ready!")

    st.markdown("---")
    st.subheader("🔍 Search Your Video")
    search_query = st.text_input("What action are you looking for?")
    
    if st.button("Search"):
        if search_query:
            if collection.count() == 0:
                st.warning("The database is empty! Please click 'Process Video' first.")
            else:
                with st.spinner(f"Searching for the best matches..."):
                    
                    # --- NATIVE CLIPFLOW SEARCH BYPASS ---
                    if selected_model == "CLIPflow (Custom Engine)":
                        text_tokens = model.clip_tokenizer([search_query]).to(model.device)
                        
                        with torch.no_grad():
                            # ✨ FIX #2: Getting the correct model data type for the search bypass ✨
                            model_dtype = model.clip_model.visual.conv1.weight.dtype
                            
                            x = model.clip_model.token_embedding(text_tokens).to(model_dtype)
                            x = x + model.clip_model.positional_embedding.to(model_dtype)
                            x = x.permute(1, 0, 2)
                            x = model.clip_model.transformer(x)
                            x = x.permute(1, 0, 2)
                            x = model.clip_model.ln_final(x)

                            text_global = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
                            if model.clip_model.text_projection is not None:
                                text_global = text_global @ model.clip_model.text_projection
                                text_words = x @ model.clip_model.text_projection
                            else:
                                text_words = x
                                
                            text_global = text_global.float()
                            text_words = text_words.float()
                            
                        results_list = []
                        # Run your pure engine on every video chunk!
                        for chunk_path, features in model.video_cache.items():
                            v_global = features["global"].unsqueeze(0).contiguous()
                            v_patches = features["patches"].unsqueeze(0).contiguous()
                            
                            # THE MAGIC HAPPENS HERE: Calling your clipflow_engine.py
                            final_score, _, _ = model.engine(v_global, v_patches, text_global, text_words)
                            
                            similarity_score = max(0, min(final_score.item() * 100, 100))
                            results_list.append((chunk_path, similarity_score))
                            
                        results_list.sort(key=lambda x: x[1], reverse=True)
                        final_results = results_list[:5] # Grab top 5
                        
                        matches_displayed = 0
                        st.subheader(f"🎯 Top Matches for: '{search_query}'")
                        for doc, score in final_results:
                            if score >= confidence_threshold:
                                st.write(f"**Match #{matches_displayed+1}** | AI Confidence Score: `{score:.2f}%`")
                                st.video(doc)
                                st.divider()
                                matches_displayed += 1

                    # --- STANDARD CHROMADB SEARCH ---
                    else:
                        query_embedding = model.get_text_embedding(search_query)
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=5,
                            include=["documents", "distances"]
                        )
                        
                        if results['documents'] and results['documents'][0]:
                            st.subheader(f"🎯 Top Matches for: '{search_query}'")
                            docs = results['documents'][0]
                            distances = results['distances'][0]
                            
                            matches_displayed = 0
                            for i, (doc, dist) in enumerate(zip(docs, distances)):
                                similarity_score = (1 - (dist / 2)) * 100
                                if similarity_score >= confidence_threshold:
                                    st.write(f"**Match #{matches_displayed+1}** | AI Confidence Score: `{similarity_score:.2f}%`")
                                    st.video(doc)
                                    st.divider()
                                    matches_displayed += 1
                                    
                    if matches_displayed == 0:
                        st.warning(f"No clips found above {confidence_threshold}%.")