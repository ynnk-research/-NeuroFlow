# > Yannick Schmitt. (2026). EMA-Gated Temporal Sequence Compression in Vision Transformers. Zenodo. https://doi.org/10.5281/zenodo.19337577
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================
# NEUROFLOW ARCHITECTURE B — VISUALIZER (MP4)
# ============================================================

import os, gc, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from transformers import AutoModel, AutoTokenizer, AutoProcessor


VIDEO_PATH       = ""
WEIGHTS_PATH     = ""
OUTPUT_MP4       = "arch_b_demo_tags.mp4"

RESOLUTION       = 448          
THRESHOLD        = 0.1
EMA_DECAY        = 0.01
N_FRAMES         = 1000          
FPS              = 25

# A competitive list of traffic scene labels
LABELS = [
    "car", "truck", "bus", "van", "scooter", "person walking", "Sidewalk", 
    "motorcycle", "bicycle", "tree", "traffic light", "street", "road", 
    "street sign", "shadow", "motorcycle", "bike", "person", "stop sign", "surface marking"
]

HAS_GPU    = torch.cuda.is_available()
DEVICE_GPU = torch.device("cuda:0") if HAS_GPU else torch.device("cpu")
DTYPE      = torch.float16 if HAS_GPU else torch.float32
# ─────────────────────────────────────────────────────────────

class VisualizerArchB(nn.Module):
    def __init__(self, base_model, threshold=0.05, ema_decay=0.01):
        super().__init__()
        self.model = base_model
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.expectation = None
        self.last_sparsity = 0.0
        self.active_mask_2d = None

    def reset_retina(self):
        self.expectation = None

    @torch.inference_mode()
    def forward(self, pixel_values):
        hidden_states = self.model.embeddings(pixel_values)
        N = hidden_states.shape[1]
        grid_1d = int(N ** 0.5)

        if self.expectation is None or self.threshold == 0.0:
            self.expectation = hidden_states.detach().clone()
            active_indices = torch.arange(N, device=pixel_values.device)
            self.active_mask_2d = np.ones((grid_1d, grid_1d), dtype=np.uint8)
        else:
            sim = F.cosine_similarity(hidden_states, self.expectation, dim=-1)
            surprise = 1.0 - sim
            active_indices = (surprise > self.threshold)[0].nonzero(as_tuple=True)[0]
            
            if len(active_indices) < 4:
                _, active_indices = torch.topk(surprise[0], 4)
            active_indices = torch.sort(active_indices)[0]
            
            self.expectation.mul_(1.0 - self.ema_decay).add_(hidden_states.detach() * self.ema_decay)
            
            mask_1d = np.zeros(N, dtype=np.uint8)
            mask_1d[active_indices.cpu().numpy()] = 1
            self.active_mask_2d = mask_1d.reshape(grid_1d, grid_1d)

        hidden_compressed = hidden_states[:, active_indices, :]
        encoder_outputs = self.model.encoder(inputs_embeds=hidden_compressed)
        sequence_output = self.model.post_layernorm(encoder_outputs[0])
        
        self.last_sparsity = 1.0 - (len(active_indices) / float(N))
        return self.model.head(sequence_output)

def upscale_siglip(base_model, target_res, patch_size=16):
    if target_res == 224: return copy.deepcopy(base_model)
    model = copy.deepcopy(base_model)
    grid_size = target_res // patch_size
    seq_len = grid_size * grid_size
    
    embed_module = model.embeddings
    old_embeds = embed_module.position_embedding.weight.data
    old_seq_len, dim = old_embeds.shape
    
    grid_embeds = old_embeds.reshape(1, int(old_seq_len**0.5), int(old_seq_len**0.5), dim).permute(0, 3, 1, 2)
    new_grid_embeds = F.interpolate(grid_embeds, size=(grid_size, grid_size), mode='bicubic', align_corners=False)
    new_embeds = new_grid_embeds.permute(0, 2, 3, 1).reshape(seq_len, dim)
        
    new_layer = nn.Embedding(seq_len, dim).to(old_embeds.device).to(old_embeds.dtype)
    new_layer.weight.data = new_embeds
    embed_module.position_embedding = new_layer
    embed_module.image_size = target_res
    embed_module.num_patches = seq_len
    embed_module.register_buffer("position_ids", torch.arange(seq_len).expand((1, -1)), persistent=False)
    return model


print(f"Loading Base SigLIP and Processor...")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
full_siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=DTYPE).to(DEVICE_GPU).eval()
base_vision = full_siglip.vision_model

# Get text embeddings
prompts = [f"a photo of a {l}" for l in LABELS]
text_inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(DEVICE_GPU)
with torch.inference_mode():
    text_embeds = full_siglip.text_model(**text_inputs).pooler_output
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
logit_scale = torch.exp(full_siglip.logit_scale).item()

if os.path.exists(WEIGHTS_PATH):
    print(f"Injecting Fine-Tuned Weights...")
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    if hasattr(state, "items"): base_vision.load_state_dict(state, strict=False)
    else: base_vision = state

print(f"Decoding Video to {RESOLUTION}p...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames_bgr, frames_tensor = [], []
while len(frames_bgr) < N_FRAMES:
    ret, bgr = cap.read()
    if not ret: break
    resized = cv2.resize(bgr, (RESOLUTION, RESOLUTION))
    frames_bgr.append(resized)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    frames_tensor.append(((t - 0.5) / 0.5).unsqueeze(0).to(DEVICE_GPU).to(DTYPE))
cap.release()

print("Initializing Models...")
dense_model = upscale_siglip(base_vision, RESOLUTION).to(DEVICE_GPU).eval()
gated_model = VisualizerArchB(upscale_siglip(base_vision, RESOLUTION).to(DEVICE_GPU), THRESHOLD, EMA_DECAY).eval()

print("Running Inference...")
telemetry = []

with torch.inference_mode():
    for _ in range(5): 
        dense_model(frames_tensor[0])
        gated_model(frames_tensor[0])
gated_model.reset_retina(); torch.cuda.synchronize()

for i, ft in enumerate(frames_tensor):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode(): emb_d = dense_model(ft).pooler_output
    torch.cuda.synchronize(); lat_d = (time.perf_counter() - t0) * 1000
    
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode(): emb_g = gated_model(ft)
    torch.cuda.synchronize(); lat_g = (time.perf_counter() - t0) * 1000

    fidelity = F.cosine_similarity(emb_g.float(), emb_d.float(), dim=-1).item() * 100
    
    # ── Semantic Tagging Logic (Top-2 Confidence) ──
    emb_d_norm = emb_d / emb_d.norm(dim=-1, keepdim=True)
    emb_g_norm = emb_g / emb_g.norm(dim=-1, keepdim=True)
    
    logits_d = (emb_d_norm @ text_embeds.T * logit_scale).squeeze(0)
    logits_g = (emb_g_norm @ text_embeds.T * logit_scale).squeeze(0)
    
    # Use softmax to get percentages
    probs_d = F.softmax(logits_d, dim=-1)
    probs_g = F.softmax(logits_g, dim=-1)
    
    top2_d = probs_d.topk(2)
    top2_g = probs_g.topk(2)
    
    tags_d = [f"{LABELS[idx]} ({prob*100:.0f}%)" for prob, idx in zip(top2_d.values, top2_d.indices)]
    tags_g = [f"{LABELS[idx]} ({prob*100:.0f}%)" for prob, idx in zip(top2_g.values, top2_g.indices)]
    
    telemetry.append({
        "lat_d": lat_d, "lat_g": lat_g, "speedup": lat_d / lat_g if lat_g > 0 else 1.0,
        "fidelity": max(0.0, fidelity), "skip": gated_model.last_sparsity * 100,
        "mask": gated_model.active_mask_2d.copy(),
        "tags_d": tags_d, "tags_g": tags_g
    })

print("Rendering MP4...")
BAR_H = 110
W, H = RESOLUTION, RESOLUTION
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, FPS, (W * 2, H + BAR_H))
font = cv2.FONT_HERSHEY_SIMPLEX

for i, bgr in enumerate(frames_bgr):
    tel = telemetry[i]
    
    # --- LEFT PANEL (Dense) ---
    panel_left = bgr.copy()
    bar_left = np.full((BAR_H, W, 3), (30, 20, 20), dtype=np.uint8) 
    cv2.putText(bar_left, f"DENSE BASELINE (100% Compute)", (10, 25), font, 0.6, (200, 200, 200), 2)
    cv2.putText(bar_left, f"Lat: {tel['lat_d']:.1f}ms", (10, 55), font, 0.55, (100, 100, 255), 2)
    cv2.putText(bar_left, f"1. {tel['tags_d'][0]}", (10, 80), font, 0.5, (255, 255, 255), 1)
    cv2.putText(bar_left, f"2. {tel['tags_d'][1]}", (10, 100), font, 0.5, (200, 200, 200), 1)
    col_left = np.vstack([bar_left, panel_left])

    # --- RIGHT PANEL (Gated Architecture B) ---
    mask_resized = cv2.resize(tel["mask"], (W, H), interpolation=cv2.INTER_NEAREST)
    mask_3d = np.stack([mask_resized]*3, axis=-1)
    panel_right = np.where(mask_3d == 1, bgr, (bgr * 0.1).astype(np.uint8))
    
    edges = cv2.Canny(mask_resized * 255, 100, 200)
    panel_right[edges > 0] = [0, 255, 0]

    bar_right = np.full((BAR_H, W, 3), (20, 30, 20), dtype=np.uint8)
    cv2.putText(bar_right, f"ARCHITECTURE B ({tel['skip']:.0f}% Skipped)", (10, 25), font, 0.6, (200, 255, 200), 2)
    cv2.putText(bar_right, f"Lat: {tel['lat_g']:.1f}ms ({tel['speedup']:.1f}x)", (10, 55), font, 0.55, (100, 255, 100), 2)
    cv2.putText(bar_right, f"Fidelity: {tel['fidelity']:.1f}%", (260, 55), font, 0.5, (255, 200, 100), 1)
    cv2.putText(bar_right, f"1. {tel['tags_g'][0]}", (10, 80), font, 0.5, (255, 255, 255), 1)
    cv2.putText(bar_right, f"2. {tel['tags_g'][1]}", (10, 100), font, 0.5, (200, 200, 200), 1)
    
    col_right = np.vstack([bar_right, panel_right])
    writer.write(np.hstack([col_left, col_right]))

writer.release()
print(f"Success! Video saved to {OUTPUT_MP4}")