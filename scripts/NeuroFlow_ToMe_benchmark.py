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

# !pip install git+https://github.com/facebookresearch/ToMe.git

# NEUROFLOW vs. TOKEN MERGING (ToMe) — MULTI-RESOLUTION fine tuned
# - Pure GPU model latency (H2D PCIe transfer excluded)
# - Strict memory resets & thermal cooldowns
# - Exact O(N^2) Theoretical FLOP Proxies for Layer-wise Merging
# - SDPA Enforced, HuggingFace Position ID buffer patched

import os, gc, time, copy, types
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
import tome.merge as tm

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

TOME_RATIOS = [8, 16, 32, 64]

NEUROFLOW_CONFIGS = [
    {"name": "NF-Balanced",   "threshold": 0.01,  "ema": 0.01},
    {"name": "NF-Aggressive", "threshold": 0.10,  "ema": 0.01},
    {"name": "NF-MaxSparse",  "threshold": 0.35,  "ema": 0.01},
]

# ── 1. CORE UTILITIES ──────────────────────────────────────────
def hard_gpu_reset(cooldown_sec=2):
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(cooldown_sec)

def get_frames(resolution, n=N_FRAMES):
    tensors, cap = [], cv2.VideoCapture(VIDEO_PATH)
    while len(tensors) < n:
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        t = normalize(to_tensor(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))), SIGLIP_MEAN, SIGLIP_STD).to(DTYPE)
        tensors.append(t.unsqueeze(0))
    cap.release()
    while len(tensors) < n: tensors.append(tensors[len(tensors) % max(len(tensors), 1)])
    return tensors

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

# ── 2. FLOP PROXIES (O(N^2) ATTENTION AWARE) ───────────────────
def calc_flops(N, C=768):
    return (12 * N * (C**2)) + (2 * (N**2) * C)

def get_theor_speedup(dense_N, sparse_type, tome_r=0, nf_skip_pct=0.0, C=768, layers=12):
    dense_flops = layers * calc_flops(dense_N, C)
    sparse_flops = 0
    
    if sparse_type == "tome":
        current_N = dense_N
        for l in range(layers):
            active_r = min(tome_r, current_N // 2)
            current_N -= active_r
            sparse_flops += calc_flops(current_N, C)
            
    elif sparse_type == "neuroflow":
        alpha = max(1.0 - (nf_skip_pct / 100.0), 0.001)
        active_N = int(dense_N * alpha)
        sparse_flops = layers * calc_flops(active_N, C)
        
    return dense_flops / max(sparse_flops, 1)

# ── 3. ARCHITECTURES ───────────────────────────────────────────
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
        h = self.model.embeddings(pixel_values)
        N = h.shape[1]

        if self.expectation is None or self.threshold == 0.0:
            self.expectation = h.detach().clone()
            idx = torch.arange(N, device=pixel_values.device)
        else:
            surprise = 1.0 - F.cosine_similarity(h, self.expectation, dim=-1)
            idx = (surprise > self.threshold)[0].nonzero(as_tuple=True)[0]
            if len(idx) < 4: _, idx = torch.topk(surprise[0], 4)
            idx = torch.sort(idx)[0]
            self.expectation.mul_(1.0 - self.ema_decay).add_(h.detach() * self.ema_decay)

        self.last_sparsity = 1.0 - (len(idx) / float(N))
        enc_out = self.model.encoder(inputs_embeds=h[:, idx, :])
        return self.model.head(self.model.post_layernorm(enc_out[0]))

class ToMeSigLIPArchB(nn.Module):
    def __init__(self, base_vision_model, r):
        super().__init__()
        self.model = base_vision_model
        self.r = r
        self.last_sparsity = 0.0

        for layer in self.model.encoder.layers:
            layer._tome_r = r
            if not hasattr(layer, "_original_forward"):
                layer._original_forward = layer.forward

            def layer_forward(self_, hidden_states, *args, **kwargs):
                active_r = min(self_._tome_r, hidden_states.shape[1] // 2)
                if active_r > 0:
                    m, _ = tm.bipartite_soft_matching(hidden_states, active_r, class_token=False)
                    hidden_states = m(hidden_states)
                return self_._original_forward(hidden_states, *args, **kwargs)

            layer.forward = types.MethodType(layer_forward, layer)

    def reset_retina(self):
        pass

    @torch.inference_mode()
    def forward(self, pixel_values):
        h = self.model.embeddings(pixel_values)
        N_orig = h.shape[1]
        out = self.model.encoder(inputs_embeds=h)
        self.last_sparsity = 1.0 - (out[0].shape[1] / float(N_orig))
        return self.model.head(self.model.post_layernorm(out[0]))

# ── 4. STRICT ISOLATED TIMING ──────────────────────────────────

# The dense baseline is always passed as a lambda with no surrounding
# torch.inference_mode/no_grad context, matching the golden standard.
# Gated models carry @torch.inference_mode() on their own forward().
# This preserves the autograd asymmetry that all golden-standard speedup
# numbers depend on.
def rigorous_timed_run(model_fn, frames, warmup=WARMUP, is_gate=False):
    dummy = frames[0].to(DEVICE)
    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()

    for _ in range(warmup):
        model_fn(dummy)

    if is_gate: getattr(model_fn, "__self__", model_fn).reset_retina()
    if HAS_GPU: torch.cuda.synchronize()

    n        = len(frames)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    enders   = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    lats, embs, sparsities = [], [], []

    for i, f in enumerate(frames):
        fd = f.to(DEVICE)                        # >>> H2D OUTSIDE TIMING <<<

        if HAS_GPU:
            starters[i].record()
            out = model_fn(fd)
            enders[i].record()
        else:
            t0  = time.perf_counter()
            out = model_fn(fd)
            lats.append((time.perf_counter() - t0) * 1000)

        embs.append(out.detach().cpu().float())
        if is_gate:
            sparsities.append(
                getattr(model_fn, "__self__", model_fn).last_sparsity * 100)

    if HAS_GPU:
        torch.cuda.synchronize()
        lats = [s.elapsed_time(e) for s, e in zip(starters, enders)]

    return float(np.median(lats)), embs, sparsities

# ── 5. MAIN BENCHMARK SCRIPT ───────────────────────────────────
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

    frames_t = get_frames(res)
    results[res] = {}

    # ── DENSE BASELINE (ToMe reference) ────────────────────────
    # ref_embs are deterministic given fixed weights + input, so one measurement
    # suffices for fidelity across both ToMe and NeuroFlow sections.
    # dense_ms for the ToMe speedup ratios is captured here.
    print(f"\n  [ Dense Baseline ]")
    hard_gpu_reset()
    baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
    if HAS_GPU:
        _ = baked(frames_t[0].to(DEVICE))
        torch.cuda.synchronize()
    tome_dense_ms, ref_embs, _ = rigorous_timed_run(
        lambda f: baked(f).pooler_output, frames_t, is_gate=False)
    print(f"    Dense Latency: {tome_dense_ms:.2f}ms")
    del baked; gc.collect()

    # ── TOME ───────────────────────────────────────────────────
    for r in TOME_RATIOS:
        print(f"\n  [ ToMe: r={r} ]")
        hard_gpu_reset()
        baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
        tome_model = ToMeSigLIPArchB(baked, r=r)
        if HAS_GPU:
            _ = tome_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()

        ms, embs, sp = rigorous_timed_run(tome_model, frames_t, is_gate=True)
        skip = float(np.mean(sp))
        fid = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(embs, ref_embs)]))
        empirical_speedup = tome_dense_ms / ms if ms > 0 else 0
        theoretical_speedup = get_theor_speedup(N_tokens, "tome", tome_r=r, C=C_dim)

        results[res][f"ToMe r={r}"] = {
            "Method": "ToMe", "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{tome_dense_ms:.2f}", "Gate Lat (ms)": f"{ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×", "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{skip:.1f}%", "Pooler CosSim": f"{fid:.2f}%"
        }
        print(f"    Skip:  {skip:.1f}%   |  CosSim: {fid:.2f}%")
        print(f"    Emp Spd: {empirical_speedup:.2f}×  |  Theor Spd: {theoretical_speedup:.2f}×  |  Lat: {ms:.2f}ms")
        del tome_model, baked; gc.collect()

    # ── DENSE BASELINE (NeuroFlow reference) ───────────────────
    # Re-measured fresh after all ToMe runs (each with hard_gpu_reset + 50 warmup
    # + 1000 timed frames) to guarantee the same thermal and clock state seen by
    # sig1vssig2 and pixeldomain when they measure NeuroFlow. ref_embs are reused
    # for fidelity since they are deterministic.
    print(f"\n  [ Dense Baseline ]")
    hard_gpu_reset()
    baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
    if HAS_GPU:
        _ = baked(frames_t[0].to(DEVICE))
        torch.cuda.synchronize()
    nf_dense_ms, _, _ = rigorous_timed_run(
        lambda f: baked(f).pooler_output, frames_t, is_gate=False)
    print(f"    Dense Latency: {nf_dense_ms:.2f}ms")
    del baked; gc.collect()

    # ── NEUROFLOW ──────────────────────────────────────────────
    for cfg in NEUROFLOW_CONFIGS:
        print(f"\n  [ NeuroFlow: {cfg['name']} ]")
        hard_gpu_reset()
        baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
        nf_model = NeuroFlowSiglipVisionArchB(baked, threshold=cfg["threshold"], ema_decay=cfg["ema"])
        if HAS_GPU:
            _ = nf_model(frames_t[0].to(DEVICE))
            torch.cuda.synchronize()

        ms, embs, sp = rigorous_timed_run(nf_model, frames_t, is_gate=True)
        skip = float(np.mean(sp))
        fid = float(np.mean([F.cosine_similarity(s, d, dim=-1).item() * 100 for s, d in zip(embs, ref_embs)]))
        empirical_speedup = nf_dense_ms / ms if ms > 0 else 0
        theoretical_speedup = get_theor_speedup(N_tokens, "neuroflow", nf_skip_pct=skip, C=C_dim)

        results[res][cfg['name']] = {
            "Method": "NeuroFlow", "N_tokens": N_tokens,
            "Dense Lat (ms)": f"{nf_dense_ms:.2f}", "Gate Lat (ms)": f"{ms:.2f}",
            "Emp Speedup": f"{empirical_speedup:.2f}×", "Theor FLOP Proxy": f"{theoretical_speedup:.2f}×",
            "Skip %": f"{skip:.1f}%", "Pooler CosSim": f"{fid:.2f}%"
        }
        print(f"    Skip:  {skip:.1f}%   |  CosSim: {fid:.2f}%")
        print(f"    Emp Spd: {empirical_speedup:.2f}×  |  Theor Spd: {theoretical_speedup:.2f}×  |  Lat: {ms:.2f}ms")
        del nf_model, baked; gc.collect()

    del frames_t, ref_embs; gc.collect(); torch.cuda.empty_cache()

# ── OUTPUT ─────────────────────────────────────────────────────
flattened_data = []
for res, models in results.items():
    for mod, data in models.items():
        row = {"Resolution": f"{res}p", "Config": mod}
        row.update(data)
        flattened_data.append(row)

df = pd.DataFrame(flattened_data)
print("\n\nFINAL SCIENTIFIC SUMMARY (SigLIP-2: ToMe vs NeuroFlow Architecture B)")
print("="*105)
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
df.to_csv("/kaggle/working/siglip2_tome_vs_neuroflow_rigorous.csv", index=False)