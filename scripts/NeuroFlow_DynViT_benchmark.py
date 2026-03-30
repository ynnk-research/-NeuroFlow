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
# NEUROFLOW vs. DynamicViT — MULTI-RESOLUTION BENCHMARK fine tuned


import os, gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from transformers import AutoModel
import pandas as pd
from tabulate import tabulate

# ─────────────────────────────────────────────────────────────
# CONFIGURATION #needs paths
# ─────────────────────────────────────────────────────────────
VIDEO_PATH      = ""
WEIGHTS_PATH    = ""

N_FRAMES     = 1000  
WARMUP       = 50   
RESOLUTIONS  = [224, 448, 896, 1792]
PATCH_SIZE   = 16

HAS_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if HAS_GPU else "cpu")
DTYPE   = torch.float16

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]

DYNAMICVIT_CONFIGS = [
    {"name": "DyViT-Light",  "keep_rates": (0.9, 0.9, 0.9)},
    {"name": "DyViT-Medium", "keep_rates": (0.7, 0.7, 0.7)},
    {"name": "DyViT-Heavy",  "keep_rates": (0.5, 0.5, 0.5)},
]

NEUROFLOW_CONFIGS = [
    {"name": "NF-Balanced",   "threshold": 0.01,  "ema": 0.01},
    {"name": "NF-Aggressive", "threshold": 0.10,  "ema": 0.01},
    {"name": "NF-MaxSparse",  "threshold": 0.35,  "ema": 0.01},
]

# ─────────────────────────────────────────────────────────────
# MEMORY-OPTIMIZED FRAME GENERATOR (Identical to multires)
# ─────────────────────────────────────────────────────────────
def get_frames(resolution, n=N_FRAMES):
    from torchvision.transforms.functional import to_tensor, normalize
    from PIL import Image
    
    tensors = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    while len(tensors) < n:
        ret, f = cap.read()
        if not ret: break
        f_resized = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB))
        t = normalize(to_tensor(img), SIGLIP_MEAN, SIGLIP_STD).to(DTYPE)
        tensors.append(t.unsqueeze(0))
    cap.release()
    
    if 0 < len(tensors) < n:
        idx = 0
        while len(tensors) < n:
            tensors.append(tensors[idx])
            idx += 1
    return tensors

# ─────────────────────────────────────────────────────────────
# POSITIONAL EMBEDDING UPSCALING (Identical to multires)
# ─────────────────────────────────────────────────────────────
import copy
def upscale_pos_embeddings(base_model, target_res, patch_size=PATCH_SIZE):
    if target_res == 224:
        return copy.deepcopy(base_model)

    model = copy.deepcopy(base_model)
    embed_module = model.embeddings
    if not hasattr(embed_module, "position_embedding"):
        raise AttributeError("Model has no position_embedding (Incompatible RoPE).")

    old_weight = embed_module.position_embedding.weight.data
    old_seq_len, dim = old_weight.shape

    old_grid = int(old_seq_len ** 0.5)
    has_cls  = (old_grid * old_grid != old_seq_len)

    if has_cls:
        cls_embed   = old_weight[0:1, :]
        grid_embeds = old_weight[1:, :]
        old_grid    = int((old_seq_len - 1) ** 0.5)
    else:
        cls_embed   = None
        grid_embeds = old_weight

    new_grid  = target_res // patch_size
    new_N     = new_grid * new_grid

    grid_4d = grid_embeds.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    new_grid_4d = F.interpolate(grid_4d, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    new_spatial = new_grid_4d.permute(0, 2, 3, 1).reshape(new_N, dim)

    new_weight  = torch.cat([cls_embed, new_spatial], dim=0) if has_cls else new_spatial
    new_seq_len = new_N + 1 if has_cls else new_N

    new_layer = nn.Embedding(new_seq_len, dim).to(old_weight.device, dtype=old_weight.dtype)
    new_layer.weight.data = new_weight
    embed_module.position_embedding = new_layer

    if hasattr(embed_module, "image_size"):
        embed_module.image_size = target_res
    if hasattr(embed_module, "num_patches"):
        embed_module.num_patches = new_N
        
    if hasattr(embed_module, "position_ids"):
        embed_module.register_buffer(
            "position_ids",
            torch.arange(new_seq_len).expand((1, -1)).to(old_weight.device),
            persistent=False
        )
    return model

# ─────────────────────────────────────────────────────────────
# NEUROFLOW GATE — Architecture B (Identical to multires)
# ─────────────────────────────────────────────────────────────
class NeuroFlowSiglipVisionArchB(nn.Module):
    def __init__(self, vision_model, threshold=0.05, ema_decay=0.01):
        super().__init__()
        self.model = vision_model
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.expectation = None
        self.last_sparsity = 0.0

    def reset_retina(self):
        self.expectation = None

    @torch.inference_mode()
    def forward(self, pixel_values):
        hidden_states = self.model.embeddings(pixel_values)
        N = hidden_states.shape[1]

        if self.expectation is None or self.threshold == 0.0:
            self.expectation = hidden_states.detach().clone()
            active_indices = torch.arange(N, device=pixel_values.device)
        else:
            surprise = 1.0 - F.cosine_similarity(hidden_states, self.expectation, dim=-1)
            active_indices = (surprise > self.threshold)[0].nonzero(as_tuple=True)[0]
            if len(active_indices) < 4:
                _, active_indices = torch.topk(surprise[0], 4)
            active_indices = torch.sort(active_indices)[0]
            self.expectation.mul_(1.0 - self.ema_decay).add_(hidden_states.detach() * self.ema_decay)

        hidden_compressed = hidden_states[:, active_indices, :]
        encoder_outputs = self.model.encoder(inputs_embeds=hidden_compressed)
        sequence_output = self.model.post_layernorm(encoder_outputs[0])
        self.last_sparsity = 1.0 - (len(active_indices) / float(N))
        return self.model.head(sequence_output)

# ─────────────────────────────────────────────────────────────
# DYNAMICVIT GATE
# ─────────────────────────────────────────────────────────────
class DynamicViTSigLIPArchB(nn.Module):
    def __init__(self, vision_model, keep_rates=(0.7, 0.7, 0.7)):
        super().__init__()
        self.model = vision_model
        self.keep_rates = keep_rates
        self.last_sparsity = 0.0
        self.prune_layers = {3, 6, 9}

    def reset_retina(self):
        # Dummy method so rigorous_timed_run can call it uniformly
        pass

    @torch.inference_mode()
    def forward(self, pixel_values):
        h = self.model.embeddings(pixel_values)
        N_orig = h.shape[1]
        p_idx = 0

        for l_idx, layer in enumerate(self.model.encoder.layers):
            h_in = h
            layer_out = layer(h_in, attention_mask=None, output_attentions=True)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if l_idx in self.prune_layers:
                N_cur = h.shape[1]
                
                # Reconstruct manual attention matrix purely for pruning importance
                norm_h = layer.layer_norm1(h_in)
                q = layer.self_attn.q_proj(norm_h)
                k = layer.self_attn.k_proj(norm_h)
                B, N_s, C = q.shape
                n_heads = layer.self_attn.num_heads
                h_dim = C // n_heads
                q = q.view(B, N_s, n_heads, h_dim).transpose(1, 2)
                k = k.view(B, N_s, n_heads, h_dim).transpose(1, 2)
                attn_w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (h_dim**0.5), dim=-1)

                imp = attn_w.mean(dim=1) 
                eye = torch.eye(N_cur, device=imp.device, dtype=torch.bool)
                imp = imp.masked_fill(eye.unsqueeze(0), 0.0) 
                score = imp[0].mean(dim=0) 

                k_keep = max(4, int(N_cur * self.keep_rates[p_idx]))
                _, top_idx = torch.topk(score, k_keep)
                top_idx = torch.sort(top_idx)[0]
                h = h[:, top_idx, :]
                p_idx += 1

        self.last_sparsity = 1.0 - (h.shape[1] / float(N_orig))
        return self.model.head(self.model.post_layernorm(h))

# ─────────────────────────────────────────────────────────────
# RIGOROUS ASYNC TIMING (Identical to multires)
# ─────────────────────────────────────────────────────────────
def rigorous_timed_run(model_fn, frames, warmup=WARMUP, is_gate=False):
    dummy = frames[0].to(DEVICE)
    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()
    
    for _ in range(warmup):
        model_fn(dummy)
        
    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()
    if HAS_GPU: torch.cuda.synchronize()

    n = len(frames)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    enders   = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    
    embs, sparsities, lats = [], [], []

    for i, f in enumerate(frames):
        fd = f.to(DEVICE)
        
        if HAS_GPU:
            starters[i].record()
            out = model_fn(fd)
            enders[i].record()
        else:
            t0 = time.perf_counter()
            out = model_fn(fd)
            lats.append((time.perf_counter() - t0) * 1000)
            
        embs.append(out.detach().cpu().float())
        if is_gate:
            sparsities.append(getattr(model_fn, "__self__", model_fn).last_sparsity * 100)

    if HAS_GPU:
        torch.cuda.synchronize()
        lats = [s.elapsed_time(e) for s, e in zip(starters, enders)]

    return float(np.median(lats)), embs, sparsities

def hard_gpu_reset(cooldown_sec=2):
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(cooldown_sec)

# ─────────────────────────────────────────────────────────────
# THEORETICAL FLOP PROXIES (O(N^2) Math)
# ─────────────────────────────────────────────────────────────
def get_theoretical_flops(N, sparse_type, C=768, layers=12, nf_skip=0.0, dy_keep_rates=None):
    dense_flops = layers * ((12 * N * (C**2)) + (2 * (N**2) * C))
    sparse_flops = 0
    
    if sparse_type == "neuroflow":
        alpha = max(1.0 - (nf_skip / 100.0), 0.001)
        active_N = int(N * alpha)
        sparse_flops = layers * ((12 * active_N * (C**2)) + (2 * (active_N**2) * C))
        
    elif sparse_type == "dyvit":
        current_N = N
        p_idx = 0
        for l in range(layers):
            sparse_flops += ((12 * current_N * (C**2)) + (2 * (current_N**2) * C))
            if l in {3, 6, 9}:
                current_N = int(current_N * dy_keep_rates[p_idx])
                p_idx += 1
                
    return dense_flops / max(sparse_flops, 1)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (Enforcing SDPA for fairness)
# ─────────────────────────────────────────────────────────────
print("Loading SigLIP 2 FixRes (Enforcing SDPA)...")
v2_full = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=DTYPE, attn_implementation="sdpa")
v2_base = v2_full.vision_model
if os.path.exists(WEIGHTS_PATH):
    state_v2 = torch.load(WEIGHTS_PATH, map_location="cpu")
    v2_base.load_state_dict(state_v2 if not hasattr(state_v2, "items") else state_v2, strict=False)

# ─────────────────────────────────────────────────────────────
# BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────
results = {}

for res in RESOLUTIONS:
    N_tokens = (res // PATCH_SIZE)**2
    C_dim = v2_base.config.hidden_size
    
    print(f"\n{'─'*85}")
    print(f"  RESOLUTION: {res}p  |  Tokens N = {N_tokens}")
    print(f"{'─'*85}")

    frames_t = get_frames(res)
    results[res] = {}

    # ── 1. Strict Reset & Wake-up for Dense ──
    print(f"\n  [ Dense Baseline ]")
    hard_gpu_reset(cooldown_sec=2)
    dense_model = upscale_pos_embeddings(v2_base, res).to(DEVICE, dtype=DTYPE).eval()
    
    if HAS_GPU:
        _ = dense_model(frames_t[0].to(DEVICE))
        torch.cuda.synchronize()

    dense_ms, dense_embs, _ = rigorous_timed_run(lambda f: dense_model(f).pooler_output, frames_t, is_gate=False)
    del dense_model
    
    print(f"    Dense Latency: {dense_ms:.2f}ms")

    # ── 2. DynamicViT Benchmark ──
    for cfg in DYNAMICVIT_CONFIGS:
        print(f"\n  [ DyViT: {cfg['name']} ]")
        hard_gpu_reset(cooldown_sec=2)
        
        # Fresh serialization mapping 1:1 with multires methodology
        dyvit_base = upscale_pos_embeddings(v2_base, res).to(DEVICE, dtype=DTYPE)
        dyvit_model = DynamicViTSigLIPArchB(dyvit_base, keep_rates=cfg["keep_rates"]).eval()
        
        if HAS_GPU:
            _ = dyvit_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()
            
        gated_ms, gated_embs, sparsities = rigorous_timed_run(dyvit_model, frames_t, is_gate=True)

        pooler_cossim = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(gated_embs, dense_embs)]))
        mean_skip = float(np.mean(sparsities))
        empirical_speedup = dense_ms / gated_ms if gated_ms > 0 else 0.0
        theoretical_speedup = get_theoretical_flops(N_tokens, "dyvit", C=C_dim, dy_keep_rates=cfg["keep_rates"])

        del dyvit_model, dyvit_base
        
        results[res][cfg['name']] = {
            "Method": "DyViT", "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{dense_ms:.2f}", "Gate Lat (ms)": f"{gated_ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×", "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{mean_skip:.1f}%", "Pooler CosSim": f"{pooler_cossim:.2f}%"
        }
        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {pooler_cossim:.2f}%")
        print(f"    Emp Spd: {empirical_speedup:.2f}×  |  Theor Spd: {theoretical_speedup:.2f}×  |  Lat: {gated_ms:.2f}ms")

    # ── 3. NeuroFlow Benchmark ──
    for cfg in NEUROFLOW_CONFIGS:
        print(f"\n  [ NeuroFlow: {cfg['name']} ]")
        hard_gpu_reset(cooldown_sec=2)
        
        # Fresh serialization mapping 1:1 with multires methodology
        nf_base = upscale_pos_embeddings(v2_base, res).to(DEVICE, dtype=DTYPE)
        nf_model = NeuroFlowSiglipVisionArchB(nf_base, threshold=cfg["threshold"], ema_decay=cfg["ema"]).eval()
        
        if HAS_GPU:
            _ = nf_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()
            
        gated_ms, gated_embs, sparsities = rigorous_timed_run(nf_model, frames_t, is_gate=True)

        pooler_cossim = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(gated_embs, dense_embs)]))
        mean_skip = float(np.mean(sparsities))
        empirical_speedup = dense_ms / gated_ms if gated_ms > 0 else 0.0
        theoretical_speedup = get_theoretical_flops(N_tokens, "neuroflow", C=C_dim, nf_skip=mean_skip)

        del nf_model, nf_base
        
        results[res][cfg['name']] = {
            "Method": "NeuroFlow", "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{dense_ms:.2f}", "Gate Lat (ms)": f"{gated_ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×", "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{mean_skip:.1f}%", "Pooler CosSim": f"{pooler_cossim:.2f}%"
        }
        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {pooler_cossim:.2f}%")
        print(f"    Emp Spd: {empirical_speedup:.2f}×  |  Theor Spd: {theoretical_speedup:.2f}×  |  Lat: {gated_ms:.2f}ms")

    del frames_t; gc.collect(); torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
flattened_data = []
for res, models in results.items():
    for mod, data in models.items():
        row = {"Resolution": f"{res}p", "Config": mod}
        row.update(data)
        flattened_data.append(row)

df = pd.DataFrame(flattened_data)
print("\n\nFINAL SCIENTIFIC SUMMARY (SigLIP-2: DyViT vs NeuroFlow)")
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))