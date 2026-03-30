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
# Pixel-domain activity vs. NeuroFlow embedding gate


import os, gc, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.transforms.functional import to_tensor, normalize
from PIL import Image
from transformers import AutoModel
import pandas as pd
from tabulate import tabulate

# ────────────────────────────────────────────────
# CONFIGURATION #needs paths
# ────────────────────────────────────────────────
VIDEO_PATH      = "needs paths"
WEIGHTS_PATH    = "needs paths"

MODEL_ID     = "google/siglip2-base-patch16-224"

N_FRAMES     = 1000
WARMUP       = 50
RESOLUTIONS  = [224, 448, 896, 1792]
PATCH_SIZE   = 16

HAS_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if HAS_GPU else "cpu")
DTYPE   = torch.float16 if HAS_GPU else torch.float32

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]

PIXEL_THRESHOLDS = [0.05, 0.15, 0.35]

NEUROFLOW_CONFIGS = [
    {"name": "NF-Balanced",   "threshold": 0.010, "ema": 0.01},
    {"name": "NF-Aggressive", "threshold": 0.100, "ema": 0.01},
    {"name": "NF-MaxSparse",  "threshold": 0.350, "ema": 0.01},
]

# ── 1. CORE UTILITIES ──────────────────────────────────────────
def hard_gpu_reset(cooldown_sec=2):
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(cooldown_sec)

def make_frames_paired(resolution, n=N_FRAMES):
    paired_data, cap = [], cv2.VideoCapture(VIDEO_PATH)
    while len(paired_data) < n:
        ret, f = cap.read()
        if not ret: break
        resized = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        t_gray = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DTYPE) / 255.0
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t_rgb = normalize(to_tensor(Image.fromarray(rgb)), SIGLIP_MEAN, SIGLIP_STD).to(DTYPE).unsqueeze(0)
        
        paired_data.append((t_rgb, t_gray))
    cap.release()
    
    while len(paired_data) < n: paired_data.append(paired_data[len(paired_data) % max(len(paired_data), 1)])
    return paired_data

def upscale_pos_embeddings(base_model, target_res, patch_size=PATCH_SIZE):
    if target_res == 224: return copy.deepcopy(base_model)
    model = copy.deepcopy(base_model)
    embed = model.embeddings
    old_weight = embed.position_embedding.weight.data
    old_seq_len, dim = old_weight.shape
    old_grid = int(old_seq_len ** 0.5)
    has_cls = (old_grid * old_grid != old_seq_len)
    
    cls_embed = old_weight[0:1, :] if has_cls else None
    grid_embeds = old_weight[1:, :] if has_cls else old_weight
    old_grid = int((old_seq_len - 1) ** 0.5) if has_cls else old_grid

    new_grid = target_res // patch_size
    new_N = new_grid * new_grid

    grid_4d = grid_embeds.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    new_grid_4d = F.interpolate(grid_4d, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    new_spatial = new_grid_4d.permute(0, 2, 3, 1).reshape(new_N, dim)

    new_weight = torch.cat([cls_embed, new_spatial], dim=0) if has_cls else new_spatial
    new_seq_len = new_N + 1 if has_cls else new_N

    new_layer = nn.Embedding(new_seq_len, dim).to(old_weight.device, dtype=old_weight.dtype)
    new_layer.weight.data = new_weight
    embed.position_embedding = new_layer

    if hasattr(embed, "image_size"): embed.image_size = target_res
    if hasattr(embed, "num_patches"): embed.num_patches = new_N
    if hasattr(embed, "position_ids"):
        embed.register_buffer("position_ids", torch.arange(new_seq_len).expand((1, -1)).to(old_weight.device), persistent=False)
    return model

# ── 2. ARCHITECTURES ───────────────────────────────────────────

class PixelActivityGateSigLIP(nn.Module):
    def __init__(self, base_vision_model, threshold=0.15, patch_size=16):
        super().__init__()
        self.model = base_vision_model
        self.threshold = threshold
        self.patch_size = patch_size
        self.prev_gray = None
        self.last_sparsity = 0.0

    def reset_retina(self):
        self.prev_gray = None

    def _patch_activity_gpu(self, gray_frame):
        _, _, H, W = gray_frame.shape
        n_patches = (H // self.patch_size) * (W // self.patch_size)

        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return torch.ones(n_patches, device=gray_frame.device, dtype=gray_frame.dtype)

        diff = torch.abs(gray_frame - self.prev_gray)
        self.prev_gray = gray_frame

        activity_grid = F.avg_pool2d(diff, kernel_size=self.patch_size, stride=self.patch_size)
        activity = activity_grid.flatten()
        
        max_act = activity.max()
        if max_act > 0: activity = activity / max_act
        return activity

    @torch.inference_mode()
    def forward(self, pixel_values, gray_frame):
        h = self.model.embeddings(pixel_values)
        N = h.shape[1]
        
        activity = self._patch_activity_gpu(gray_frame)
        idx = (activity > self.threshold).nonzero(as_tuple=True)[0]

        if len(idx) < 4: _, idx = torch.topk(activity, 4)
        idx = torch.sort(idx)[0]

        self.last_sparsity = 1.0 - (len(idx) / float(N))
        enc_out = self.model.encoder(inputs_embeds=h[:, idx, :])
        return self.model.head(self.model.post_layernorm(enc_out[0]))

# BIT-FOR-BIT IDENTICAL to sig1vssig2_multires.py
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

# ── 3. STRICT ISOLATED TIMING (Identical to sig1vssig2) ────────
def rigorous_timed_run(model_fn, frames, warmup=WARMUP, is_gate=False, state_model=None):
    # state_model allows lambda-wrapped models (e.g. NeuroFlow called with gray discarded)
    # to still have reset_retina() and last_sparsity resolved against the true nn.Module.
    # When model_fn IS the nn.Module directly, state_model defaults to model_fn itself.
    model_ref = state_model if state_model is not None else getattr(model_fn, "__self__", model_fn)

    dummy_rgb, dummy_gray = frames[0][0].to(DEVICE), frames[0][1].to(DEVICE)
    
    # 1. Warmup
    if is_gate: model_ref.reset_retina()
    for _ in range(warmup):
        model_fn(dummy_rgb, dummy_gray)
        
    if is_gate: model_ref.reset_retina()
    if HAS_GPU: torch.cuda.synchronize()

    # 2. Setup Events
    n = len(frames)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    enders   = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    
    embs, sparsities, lats = [], [], []

    # 3. Hot Loop (Pure Dispatch, No Syncing, H2D isolated)
    for i, (rgb, gray) in enumerate(frames):
        # >>> HOST TO DEVICE OUTSIDE TIMING <<<
        rgb_d  = rgb.to(DEVICE)
        gray_d = gray.to(DEVICE)
        
        if HAS_GPU:
            starters[i].record()
            out = model_fn(rgb_d, gray_d)
            enders[i].record()
        else:
            t0 = time.perf_counter()
            out = model_fn(rgb_d, gray_d)
            lats.append((time.perf_counter() - t0) * 1000)
            
        embs.append(out.detach().cpu().float())
        if is_gate:
            sparsities.append(model_ref.last_sparsity * 100.0)

    # 4. Global Sync
    if HAS_GPU:
        torch.cuda.synchronize()
        lats = [s.elapsed_time(e) for s, e in zip(starters, enders)]

    return float(np.median(lats)), embs, sparsities

# ── 4. MAIN BENCHMARK SCRIPT ───────────────────────────────────
print(f"Loading {MODEL_ID} (Enforcing SDPA)...")
v_full = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
base_vision = v_full.vision_model

if os.path.exists(WEIGHTS_PATH):
    st = torch.load(WEIGHTS_PATH, map_location="cpu")
    base_vision.load_state_dict(st if not hasattr(st, "items") else st, strict=False)

results = {}

for res in RESOLUTIONS:
    N_tokens = (res // PATCH_SIZE) ** 2
    C_dim = base_vision.config.hidden_size
    print(f"\n{'─'*95}\n  RESOLUTION: {res}p  |  Tokens N = {N_tokens}\n{'─'*95}")

    paired_frames = make_frames_paired(res)
    results[res] = {}

    # ── DENSE ──
    print(f"\n  [ Dense Baseline ]")
    hard_gpu_reset()
    dense_model = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
    
    # Wake up
    if HAS_GPU:
        _ = dense_model(paired_frames[0][0].to(DEVICE))
        torch.cuda.synchronize()
    
    dense_ms, dense_embs, _ = rigorous_timed_run(lambda rgb, gray: dense_model(rgb).pooler_output, paired_frames, is_gate=False)
    print(f"    Dense Latency: {dense_ms:.2f}ms")
    del dense_model; gc.collect()

    # ── PIXEL-DOMAIN GATE ──
    for thresh in PIXEL_THRESHOLDS:
        print(f"\n  [ Pixel Gate: θ={thresh} ]")
        hard_gpu_reset()
        baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
        pg_model = PixelActivityGateSigLIP(baked, threshold=thresh, patch_size=PATCH_SIZE)
        
        if HAS_GPU:
            _ = pg_model(paired_frames[0][0].to(DEVICE), paired_frames[0][1].to(DEVICE))
            torch.cuda.synchronize()

        gated_ms, gated_embs, sparsities = rigorous_timed_run(pg_model, paired_frames, is_gate=True)
        
        pooler_cossim = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(gated_embs, dense_embs)]))
        mean_skip = float(np.mean(sparsities))
        empirical_speedup = dense_ms / gated_ms if gated_ms > 0 else 0.0
        
        # Exact O(N^2) Math
        alpha = max(1.0 - (mean_skip / 100.0), 0.001)
        dense_flops  = 12 * ((12 * N_tokens * (C_dim**2)) + (2 * (N_tokens**2) * C_dim))
        sparse_flops = 12 * ((12 * (alpha * N_tokens) * (C_dim**2)) + (2 * ((alpha * N_tokens)**2) * C_dim))
        theoretical_speedup = dense_flops / sparse_flops

        results[res][f"Pixel Gate θ={thresh}"] = {
            "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{dense_ms:.2f}",
            "Gate Lat (ms)": f"{gated_ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×",
            "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{mean_skip:.1f}%",
            "Pooler CosSim": f"{pooler_cossim:.2f}%"
        }
        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {pooler_cossim:.2f}%")
        print(f"    Empirical Speedup: {empirical_speedup:.2f}×  |  Theoretical: {theoretical_speedup:.2f}×")
        del pg_model, baked; gc.collect()

    # ── NEUROFLOW (Embedding Gate) ──
    for cfg in NEUROFLOW_CONFIGS:
        print(f"\n  [ NeuroFlow: {cfg['name']} ]")
        hard_gpu_reset()
        baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
        nf_model = NeuroFlowSiglipVisionArchB(baked, threshold=cfg["threshold"], ema_decay=cfg["ema"])
        
        if HAS_GPU:
            _ = nf_model(paired_frames[0][0].to(DEVICE))
            torch.cuda.synchronize()

        # Lambda cleanly discards the gray tensor to keep identical memory boundary.
        # state_model=nf_model ensures reset_retina() and last_sparsity are resolved
        # against the true nn.Module, not the lambda (which has no __self__).
        gated_ms, gated_embs, sparsities = rigorous_timed_run(
            lambda rgb, gray: nf_model(rgb), paired_frames, is_gate=True, state_model=nf_model)
        
        pooler_cossim = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(gated_embs, dense_embs)]))
        mean_skip = float(np.mean(sparsities))
        empirical_speedup = dense_ms / gated_ms if gated_ms > 0 else 0.0
        
        # Exact O(N^2) Math
        alpha = max(1.0 - (mean_skip / 100.0), 0.001)
        dense_flops  = 12 * ((12 * N_tokens * (C_dim**2)) + (2 * (N_tokens**2) * C_dim))
        sparse_flops = 12 * ((12 * (alpha * N_tokens) * (C_dim**2)) + (2 * ((alpha * N_tokens)**2) * C_dim))
        theoretical_speedup = dense_flops / sparse_flops

        results[res][f"NeuroFlow {cfg['name']}"] = {
            "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{dense_ms:.2f}",
            "Gate Lat (ms)": f"{gated_ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×",
            "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{mean_skip:.1f}%",
            "Pooler CosSim": f"{pooler_cossim:.2f}%"
        }
        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {pooler_cossim:.2f}%")
        print(f"    Empirical Speedup: {empirical_speedup:.2f}×  |  Theoretical: {theoretical_speedup:.2f}×")
        del nf_model, baked; gc.collect()

    del paired_frames, dense_embs; gc.collect(); torch.cuda.empty_cache()

# ── OUTPUT ─────────────────────────────────────────────────────
flattened_data = []
for res, models in results.items():
    for mod, data in models.items():
        row = {"Resolution": f"{res}p", "Method": mod}
        row.update(data)
        flattened_data.append(row)

df = pd.DataFrame(flattened_data)
print("\n\nFINAL SCIENTIFIC SUMMARY (Ablation: Pixel Gate vs NeuroFlow Architecture B)")
print("="*105)
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))