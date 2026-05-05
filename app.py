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


class ClipperCustomModel(BaseAIModel):
    """Your proprietary local model using the Clipper architecture."""
    def __init__(self):
        import torch
        # Import your local custom Python files!
        try:
            from config import ClipperConfig
            from model import ClipperModel
        except ImportError:
            st.error("Missing config.py or model.py! Please put them in the same folder as app.py.")
            st.stop()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = "clipper_best.pt" # Ensure this file is in your folder!
        
        # Load your custom architecture
        self.config = ClipperConfig()
        self.config.use_custom_modules = True
        self.model = ClipperModel(self.config).eval().to(self.device)
        
        # Load your custom weights
        if os.path.exists(self.weights_path):
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)
        else:
            st.warning(f"Weights not found at {self.weights_path}. Running zero-shot mode.")

    def get_video_embedding(self, video_path):
        import cv2
        from PIL import Image
        import torch
        
        # We use your exact custom frame sampling logic via OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        idxs = np.linspace(0, max(0, total_f - 1), self.config.num_frames, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        
        while len(frames) < self.config.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        frames = frames[:self.config.num_frames]
        
        imgs = [self.model.preprocess(Image.fromarray(f)) for f in frames]
        x = torch.stack(imgs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get your custom embeddings
            clip_e, frame_e = self.model.encode_video(x)
            
        # We return the primary global clip embedding for ChromaDB
        return clip_e.squeeze().tolist()

    def get_text_embedding(self, text):
        import torch
        with torch.no_grad():
            tokens = self.model.tokenizer([text]).to(self.device)
            t_emb, c_emb = self.model.encode_text(tokens)
            
        # Return the primary text embedding
        return t_emb.squeeze().tolist()

class ClipAdvancedModel(BaseAIModel):
    """CLIP Advanced Model Wrapper"""
    def __init__(self):
        import sys
        import os
        from types import SimpleNamespace
        from transformers import CLIPProcessor
        
        # Ensure folder is named 'final_model' with an underscore
        sys.path.append(os.path.join(os.path.dirname(__file__), 'final_model'))
        try:
            from final_model import modeling
        except ImportError:
            st.error("Could not import final_model.modeling. Check your folder structure.")
            st.stop()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.config = SimpleNamespace(
            interaction='wti',
            agg_module='seqTransf',
            base_encoder='ViT-B/32',
            center=1,
            query_number=8,
            cross_att_layer=3,
            num_hidden_layers=4,
            temp=3.0,
            loss2_weight=0.5,
            alpha=0.0001,
            beta=0.005,
            query_share=True,
            cross_att_share=True,
            max_frames=12,
            max_words=32
        )
        
        # 🚨 THE MISSING LINE: We must create the empty model BEFORE we load weights!
        self.model = modeling.DiCoSA(self.config).to(self.device).eval()
        
        # Now we locate and load your trained Kaggle weights
        self.weights_path = "pytorch_model.bin.best.2"
        
        if os.path.exists(self.weights_path):
            ckpt = torch.load(self.weights_path, map_location=self.device)
            
            # Strip the "module." prefix from the Kaggle multi-GPU weights
            unwrapped_ckpt = {}
            for k, v in ckpt.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                unwrapped_ckpt[new_k] = v
                
            # Load the cleaned weights into the model
            missing, unexpected = self.model.load_state_dict(unwrapped_ckpt, strict=False)
            print(f"✅ SUCCESSFULLY LOADED GLSCL BRAIN: {self.weights_path}")
            
            # This will warn us if the weights still aren't mapping correctly
            if len(missing) > 0:
                print(f"⚠️ Warning: Missing keys detected: {missing[:5]}...")
        else:
            print(f"❌ ERROR: Could not find {self.weights_path}. Model will use random weights.")
            
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.video_cache = {}
        
    def get_video_embedding(self, video_path):
        import cv2
        import torch
        from PIL import Image
        
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        num_frames = getattr(self.config, 'max_frames', 12)
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
        
        inputs = self.processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device) 
        
        video_tensor = pixel_values.unsqueeze(0) 
        video_mask = torch.ones((1, len(frames))).to(self.device)
        
        with torch.no_grad():
            video_feat = self.model.get_video_feat(video_tensor, video_mask)
            
        # FIX: Store the true, un-averaged tensors in memory!
        self.video_cache[video_path] = {
            "video_feat": video_feat.cpu(),
            "video_mask": video_mask.cpu()
        }
            
        # Dummy return for ChromaDB
        video_emb = video_feat.mean(dim=1).squeeze()
        video_emb = F.normalize(video_emb, p=2, dim=-1)
        return video_emb.tolist()

    def get_text_embedding(self, text):
        return np.zeros(512).tolist()

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

        # ✨ BUG 2 FIX: Add the missing Post-LayerNorm! ✨
        x = visual.ln_post(x)

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
        
        # ✨ BUG 1 FIX: Return actual normalized embeddings instead of zeros
        vid_emb = video_global.mean(dim=0)
        vid_emb = F.normalize(vid_emb, p=2, dim=-1)
        return vid_emb.tolist() 

    def get_text_embedding(self, text):
        # ✨ BUG 1 FIX: Return actual normalized embeddings instead of zeros
        text_tokens = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features[0].tolist()
    
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
        "CLIP Advanced",
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
def load_final_model(): return ClipAdvancedModel()

if selected_model == "Original CLIP (Baseline)": model = load_baseline()
elif selected_model == "Searchium CLIP4Clip (High Accuracy)": model = load_clip4clip()
elif selected_model == "Clipper (Custom Local Model)": model = load_clipper()
elif selected_model == "CLIPflow (Custom Engine)": model = load_clipflow()
elif selected_model == "CLIP Advanced": model = load_final_model()
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
                    
                    # --- NATIVE ENGINE SEARCH BYPASS ---
                    if selected_model in ["CLIPflow (Custom Engine)", "CLIP Advanced"]:
                        results_list = []
                        
                        if selected_model == "CLIPflow (Custom Engine)":
                            text_tokens = model.clip_tokenizer([search_query]).to(model.device)
                            with torch.no_grad():
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
                                
                            for chunk_path, features in model.video_cache.items():
                                v_global = features["global"].unsqueeze(0).contiguous().to(model.device)
                                v_patches = features["patches"].unsqueeze(0).contiguous().to(model.device)
                                final_score, _, _ = model.engine(v_global, v_patches, text_global, text_words)
                                
                                raw_score = final_score.item()
                                if raw_score > 1.0: raw_score = raw_score / 100.0 
                                similarity_score = max(0.0, min(raw_score * 100.0, 100.0))
                                results_list.append((chunk_path, similarity_score))
                                
                        elif selected_model == "CLIP Advanced":
                            inputs = model.processor(text=[search_query], return_tensors="pt", padding="max_length", truncation=True, max_length=32)
                            text_ids = inputs["input_ids"].to(model.device)
                            text_mask = inputs["attention_mask"].to(model.device)
                            
                            with torch.no_grad():
                                text_feat, cls_feat = model.model.get_text_feat(text_ids, text_mask)
                                
                            for chunk_path, features in model.video_cache.items():
                                video_feat = features["video_feat"].to(model.device)
                                video_mask = features["video_mask"].to(model.device)
                                
                                with torch.no_grad():
                                    retrieve_logits, _, _, _ = model.model.similarity(text_feat, cls_feat, video_feat, text_mask, video_mask)
                                    
                                # ✨ FIX 2: Use ONLY the pure, refined Cross-Modal retrieval score!
                                raw_score = retrieve_logits.item()
                                
                                # Convert pure cosine similarity to percentage
                                similarity_score = ((raw_score + 1.0) / 2.0) * 100.0
                                similarity_score = max(0.0, min(similarity_score, 100.0))
                                results_list.append((chunk_path, similarity_score))

                        results_list.sort(key=lambda x: x[1], reverse=True)
                        final_results = results_list[:5] 
                        
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