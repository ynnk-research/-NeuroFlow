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
# NEUROFLOW — ARCHITECTURE B MULTI-RESOLUTION BENCHMARK
# RIGOROUS SCIENTIFIC EVALUATION (SigLIP v1 vs v2)
#
# FIX 1: CUDA Events used for timing without CPU blocking.
# FIX 2: Attention kernels strictly matched (SDPA).
# FIX 3: Memory processed safely to prevent OS Swap-lag.
# FIX 4: Theoretical FLOP proxies included.
# ============================================================

import os, gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from transformers import AutoModel
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIGURATION 
# ─────────────────────────────────────────────────────────────
VIDEO_PATH      = ""
WEIGHTS_PATH_V1    = ""
WEIGHTS_PATH_V2    = ""

N_FRAMES     = 1000  
WARMUP       = 50   
RESOLUTIONS  = [224, 448, 896, 1792]
THRESHOLD    = 0.35
EMA_DECAY    = 0.01
PATCH_SIZE   = 16

HAS_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if HAS_GPU else "cpu")
DTYPE   = torch.float16

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]


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
    
    # Pad if short
    if 0 < len(tensors) < n:
        idx = 0
        while len(tensors) < n:
            tensors.append(tensors[idx])
            idx += 1
    return tensors

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
        
    # Overwrite the cached position_ids to match the new 784/3136/12544 sequence length
    if hasattr(embed_module, "position_ids"):
        embed_module.register_buffer(
            "position_ids",
            torch.arange(new_seq_len).expand((1, -1)).to(old_weight.device),
            persistent=False
        )


    return model

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
# RIGOROUS ASYNC TIMING (NO PCIe BLOCKING)
# ─────────────────────────────────────────────────────────────
def rigorous_timed_run(model_fn, frames, warmup=WARMUP, is_gate=False):
    dummy = frames[0].to(DEVICE)
    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()
    
    # 1. Warmup (stabilizes cuDNN workspaces)
    for _ in range(warmup):
        model_fn(dummy)
        
    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()
    if HAS_GPU: torch.cuda.synchronize()

    # 2. Allocate Events
    n = len(frames)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    enders   = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    
    embs, sparsities, lats = [], [], []

    # 3. Hot Loop (Pure Dispatch, No Syncing)
    for i, f in enumerate(frames):
        fd = f.to(DEVICE)
        
        if HAS_GPU:
            starters[i].record()
            out = model_fn(fd)
            enders[i].record()
        else:
            import time
            t0 = time.perf_counter()
            out = model_fn(fd)
            lats.append((time.perf_counter() - t0) * 1000)
            
        embs.append(out.detach().cpu().float())
        if is_gate:
            sparsities.append(getattr(model_fn, "__self__", model_fn).last_sparsity * 100)

    # 4. Global Sync and Time Calculation
    if HAS_GPU:
        torch.cuda.synchronize()
        lats = [s.elapsed_time(e) for s, e in zip(starters, enders)]

    return float(np.median(lats)), embs, sparsities

print("Loading SigLIP v1 (Enforcing SDPA)...")
v1_full = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=DTYPE, attn_implementation="sdpa")
v1_base = v1_full.vision_model
if os.path.exists(WEIGHTS_PATH_V1):
    state = torch.load(WEIGHTS_PATH_V1, map_location="cpu")
    v1_base.load_state_dict(state if not hasattr(state, "items") else state, strict=False)

print("Loading SigLIP 2 FixRes (Enforcing SDPA)...")
try:
    v2_full = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=DTYPE, attn_implementation="sdpa")
    v2_base = v2_full.vision_model
    if os.path.exists(WEIGHTS_PATH_V2):
        state_v2 = torch.load(WEIGHTS_PATH_V2, map_location="cpu")
        v2_base.load_state_dict(state_v2 if not hasattr(state_v2, "items") else state_v2, strict=False)
    HAS_V2 = True
except Exception as e:
    HAS_V2 = False

models_to_run = [("v1_finetuned", v1_base)]
if HAS_V2: models_to_run.append(("v2_finetuned", v2_base))

def hard_gpu_reset(cooldown_sec=2):
    """
    Nukes memory fragmentation, clears Inter-Process Communication (IPC) metadata,
    and allows the GPU dynamic clocks (boost/thermal) to return to a baseline state.
    """
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(cooldown_sec)
results = {}

for res in RESOLUTIONS:
    print(f"\n{'─'*75}")
    print(f"  RESOLUTION: {res}p  |  Tokens N = {(res//PATCH_SIZE)**2}")
    print(f"{'─'*75}")

    frames_t = get_frames(res)
    results[res] = {}

    for model_name, base_model in models_to_run:
        print(f"\n  [{model_name}]")
        
        # ── 1. Strict Reset & Wake-up for Dense ──
        hard_gpu_reset(cooldown_sec=2)
        dense_model = upscale_pos_embeddings(base_model, res).to(DEVICE, dtype=DTYPE).eval()
        
        # Wake up GPU from idle power state before starting the formal warmup/timing
        if HAS_GPU:
            _ = dense_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()

        dense_ms, dense_embs, _ = rigorous_timed_run(lambda f: dense_model(f).pooler_output, frames_t, is_gate=False)
        
        # Purge Dense Model completely
        del dense_model
        
        # ── 2. Strict Reset & Wake-up for Sparse ──
        hard_gpu_reset(cooldown_sec=2)
        gated_base = upscale_pos_embeddings(base_model, res).to(DEVICE, dtype=DTYPE)
        gate_model = NeuroFlowSiglipVisionArchB(gated_base, threshold=THRESHOLD, ema_decay=EMA_DECAY).eval()
        
        # Wake up GPU from idle power state
        if HAS_GPU:
            _ = gate_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()
            
        gated_ms, gated_embs, sparsities = rigorous_timed_run(gate_model, frames_t, is_gate=True)

        # ── Scientific Metrics Calculation ──
        pooler_cossim = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(gated_embs, dense_embs)]))
        mean_skip = float(np.mean(sparsities))
        
        empirical_speedup = dense_ms / gated_ms if gated_ms > 0 else 0.0
        
        # RIGOROUS THEORETICAL FLOP PROXY (Accounts for O(N^2) Attention math)
        alpha = max(1.0 - (mean_skip / 100.0), 0.001)
        C = base_model.config.hidden_size 
        N_tokens = (res // PATCH_SIZE) ** 2
        
        dense_flops  = (12 * N_tokens * (C**2)) + (2 * (N_tokens**2) * C)
        sparse_flops = (12 * (alpha * N_tokens) * (C**2)) + (2 * ((alpha * N_tokens)**2) * C)
        theoretical_speedup = dense_flops / sparse_flops

        # Purge Sparse Model completely
        del gate_model, gated_base
        
        results[res][model_name] = {
            "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{dense_ms:.2f}",
            "Gate Lat (ms)": f"{gated_ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×",
            "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{mean_skip:.1f}%",
            "Pooler CosSim": f"{pooler_cossim:.2f}%"
        }

        print(f"    Dense: {dense_ms:.2f}ms  |  Gated: {gated_ms:.2f}ms")
        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {pooler_cossim:.2f}%")
        print(f"    Empirical Speedup: {empirical_speedup:.2f}×  |  Theoretical: {theoretical_speedup:.2f}×")

    del frames_t; gc.collect(); torch.cuda.empty_cache()


flattened_data = []
for res, models in results.items():
    for mod, data in models.items():
        row = {"Resolution": f"{res}p", "Model": mod}
        row.update(data)
        flattened_data.append(row)

df = pd.DataFrame(flattened_data)
print("\n\nFINAL SCIENTIFIC SUMMARY (Architecture B)")
from tabulate import tabulate
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

DARK_BG  = "#0f0f0f"
PANEL_BG = "#1a1a2e"
COL_V1   = "#00FF88"   # v1 fine-tuned — green
COL_V2   = "#A78BFA"   # v2 fine-tuned — purple

fig = plt.figure(figsize=(20, 10), facecolor=DARK_BG)
fig.suptitle(
    "NeuroFlow Architecture B — Multi-Resolution Benchmark\n"
    "SigLIP v1 (fine-tuned) vs SigLIP 2 FixRes (fine-tuned)",
    color="white", fontsize=14)

gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
res_labels = [f"{r}p\n(N={int((r/PATCH_SIZE)**2)})" for r in RESOLUTIONS]
x  = np.arange(len(RESOLUTIONS))
w  = 0.35

def get_series(metric, model_name):
    return [results[r][model_name][metric]
            for r in RESOLUTIONS if model_name in results.get(r, {})]

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_xticks(x); ax.set_xticklabels(res_labels, color="white", fontsize=8)

# Panel 1: Dense latency
ax1 = fig.add_subplot(gs[0, 0])
v1_dense = get_series("dense_ms", "v1_finetuned")
v2_dense = get_series("dense_ms", "v2_finetuned") if HAS_V2 else []
ax1.bar(x - w/2, v1_dense, w, label="SigLIP v1 (FT) dense", color=COL_V1, alpha=0.7)
if v2_dense:
    ax1.bar(x + w/2, v2_dense, w, label="SigLIP v2 (FT) dense", color=COL_V2, alpha=0.7)
ax1.set_ylabel("Latency (ms)", color="white")
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}ms"))
style_ax(ax1, "Dense Baseline Latency (log scale)")
ax1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

# Panel 2: Speedup
ax2 = fig.add_subplot(gs[0, 1])
v1_spdup = get_series("speedup", "v1_finetuned")
v2_spdup = get_series("speedup", "v2_finetuned") if HAS_V2 else []
ax2.plot(x, v1_spdup, "o-", color=COL_V1, linewidth=2.5, markersize=8, label="v1 FT speedup")
if v2_spdup:
    ax2.plot(x, v2_spdup, "s--", color=COL_V2, linewidth=2.5, markersize=8, label="v2 FT speedup")
ax2.axhline(1.0, color="white", linewidth=0.8, linestyle=":", alpha=0.5)
for i, (v, label) in enumerate(zip(v1_spdup, res_labels)):
    ax2.annotate(f"{v:.1f}×", (i, v), textcoords="offset points",
                 xytext=(0, 8), ha="center", color=COL_V1, fontsize=8)
ax2.set_ylabel("Speedup ×", color="white")
style_ax(ax2, "NeuroFlow Speedup vs Resolution")
ax2.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

# Panel 3: Fidelity
ax3 = fig.add_subplot(gs[1, 0])
v1_fid = get_series("fidelity", "v1_finetuned")
v2_fid = get_series("fidelity", "v2_finetuned") if HAS_V2 else []
ax3.plot(x, v1_fid, "o-", color=COL_V1, linewidth=2.5, markersize=8, label="v1 FT fidelity")
if v2_fid:
    ax3.plot(x, v2_fid, "s--", color=COL_V2, linewidth=2.5, markersize=8, label="v2 FT fidelity")
ax3.axhline(95, color="#00FF88", linewidth=1.5, linestyle="--", alpha=0.6, label="95% threshold")
ax3.axhline(85, color="#FFD700", linewidth=1.2, linestyle=":", alpha=0.6, label="85% threshold")
ax3.set_ylim(0, 105)
ax3.set_ylabel("Embedding Fidelity %", color="white")
style_ax(ax3, "Fidelity vs Dense Baseline")
ax3.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

# Panel 4: Skip % and N_active tokens
ax4 = fig.add_subplot(gs[1, 1])
v1_skip = get_series("skip_pct", "v1_finetuned")
v2_skip = get_series("skip_pct", "v2_finetuned") if HAS_V2 else []
ax4.bar(x - w/2, v1_skip, w, label="v1 FT skip %", color=COL_V1, alpha=0.7)
if v2_skip:
    ax4.bar(x + w/2, v2_skip, w, label="v2 FT skip %", color=COL_V2, alpha=0.7)

# Overlay active token count on secondary axis
ax4b = ax4.twinx()
n_tokens = [(r // PATCH_SIZE) ** 2 for r in RESOLUTIONS]
v1_active = [n * (1 - s/100) for n, s in zip(n_tokens, v1_skip)]
ax4b.plot(x, v1_active, "o-", color=COL_V1, linewidth=2, markersize=6,
          linestyle=":", label="v1 active tokens")
if v2_skip:
    v2_active = [n * (1 - s/100) for n, s in zip(n_tokens, v2_skip)]
    ax4b.plot(x, v2_active, "s-", color=COL_V2, linewidth=2, markersize=6,
              linestyle=":", label="v2 active tokens")
ax4b.set_ylabel("Active tokens (N_active)", color="white", fontsize=9)
ax4b.tick_params(colors="white", axis="y")

ax4.set_ylabel("Skip %", color="white")
ax4.set_ylim(0, 105)
style_ax(ax4, "Sparsity & Active Token Count vs Resolution")
lines1, labs1 = ax4.get_legend_handles_labels()
lines2, labs2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labs1 + labs2,
           facecolor=PANEL_BG, labelcolor="white", fontsize=8, loc="lower right")

plt.savefig("/kaggle/working/archb_multires_v1_v2.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print("\nSaved: archb_multires_v1_v2.png")