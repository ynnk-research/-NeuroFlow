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
# IQR LATENCY DISTRIBUTION — KEY CONFIG CLAIMS
#
# Timing  : Pure GPU via CUDA Events (H2D outside boundary)
# Resets  : hard_gpu_reset() before every model instantiation
# Warmup  : reset_retina() before warmup AND before timed run
# SDPA    : enforced via attn_implementation="sdpa"
# Reports : median, P25, P75, IQR, P5, P95, skip%, theoretical
#           FLOP proxy, empirical speedup with IQR-derived bounds
#
# Configs : Dense (all resolutions)
#           NF-Aggressive θ=0.10 (all resolutions)
#           NF-MaxSparse  θ=0.35 (all resolutions)


import gc, time, copy, os
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

# ── # CONFIGURATION #needs paths ──────────────────────────────────────────────
MODEL_ID     = "google/siglip2-base-patch16-224"
VIDEO_PATH      = "needs paths"
WEIGHTS_PATH    = "needs paths"


N_FRAMES    = 1000
WARMUP      = 50
RESOLUTIONS = [224, 448, 896, 1792]
PATCH_SIZE  = 16

HAS_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if HAS_GPU else "cpu")
DTYPE   = torch.float16 if HAS_GPU else torch.float32

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]

NEUROFLOW_CONFIGS = [
    {"name": "NF-Balanced", "threshold": 0.010, "ema": 0.01},
    {"name": "NF-Aggressive", "threshold": 0.10, "ema": 0.01},
    {"name": "NF-MaxSparse",  "threshold": 0.35, "ema": 0.01},
]

def hard_gpu_reset(cooldown_sec=2):
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(cooldown_sec)

def calc_flops(N, C=768):
    return (12 * N * (C**2)) + (2 * (N**2) * C)

def get_theor_speedup(dense_N, nf_skip_pct, C=768, layers=12):
    dense_flops  = layers * calc_flops(dense_N, C)
    alpha        = max(1.0 - (nf_skip_pct / 100.0), 0.001)
    sparse_flops = layers * calc_flops(int(dense_N * alpha), C)
    return dense_flops / sparse_flops

def stats(arr):
    return {
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "iqr":    float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "p05":    float(np.percentile(arr,  5)),
        "p95":    float(np.percentile(arr, 95)),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
    }

def upscale_pos_embeddings(base_model, target_res, patch_size=PATCH_SIZE):
    if target_res == 224:
        return copy.deepcopy(base_model)
    model      = copy.deepcopy(base_model)
    embed      = model.embeddings
    old_weight = embed.position_embedding.weight.data
    old_seq_len, dim = old_weight.shape
    old_grid   = int(old_seq_len ** 0.5)
    has_cls    = (old_grid * old_grid != old_seq_len)

    cls_embed   = old_weight[0:1, :] if has_cls else None
    grid_embeds = old_weight[1:, :]  if has_cls else old_weight
    old_grid    = int((old_seq_len - 1) ** 0.5) if has_cls else old_grid

    new_grid    = target_res // patch_size
    new_N       = new_grid * new_grid
    grid_4d     = grid_embeds.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    new_grid_4d = F.interpolate(grid_4d, size=(new_grid, new_grid),
                                mode="bicubic", align_corners=False)
    new_spatial = new_grid_4d.permute(0, 2, 3, 1).reshape(new_N, dim)

    new_weight  = torch.cat([cls_embed, new_spatial], dim=0) if has_cls else new_spatial
    new_seq_len = new_N + 1 if has_cls else new_N
    new_layer   = nn.Embedding(new_seq_len, dim).to(old_weight.device, dtype=old_weight.dtype)
    new_layer.weight.data = new_weight
    embed.position_embedding = new_layer

    if hasattr(embed, "image_size"):   embed.image_size   = target_res
    if hasattr(embed, "num_patches"):  embed.num_patches  = new_N
    if hasattr(embed, "position_ids"):
        embed.register_buffer(
            "position_ids",
            torch.arange(new_seq_len).expand((1, -1)).to(old_weight.device),
            persistent=False)
    return model

# ── 3. FRAME LOADING (CPU tensors — H2D happens in timing harness) ──
def make_frames(resolution, n=N_FRAMES):
    tensors = []
    cap     = cv2.VideoCapture(VIDEO_PATH)
    while len(tensors) < n:
        ret, f = cap.read()
        if not ret: break
        resized = cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        img     = Image.fromarray(resized[..., ::-1].copy())
        t       = normalize(to_tensor(img), SIGLIP_MEAN, SIGLIP_STD)
        tensors.append(t.unsqueeze(0))                           # stays on CPU
    cap.release()
    while len(tensors) < n:
        tensors.append(tensors[len(tensors) % max(len(tensors), 1)])
    return tensors[:n]

class DenseSigLIP(nn.Module):
    def __init__(self, base_vision_model):
        super().__init__()
        self.model        = base_vision_model
        self.last_sparsity = 0.0

    def reset_retina(self): pass

    def forward(self, pixel_values):
        return self.model(pixel_values).pooler_output


class NeuroFlowSigLIP(nn.Module):
    def __init__(self, base_vision_model, threshold=0.05, ema_decay=0.01):
        super().__init__()
        self.model        = base_vision_model
        self.threshold    = threshold
        self.ema_decay    = ema_decay
        self.expectation  = None
        self.last_sparsity = 0.0

    def reset_retina(self):
        self.expectation = None

    @torch.inference_mode()
    def forward(self, pixel_values):
        h = self.model.embeddings(pixel_values)
        N = h.shape[1]

        if self.expectation is None:
            self.expectation = h.detach().clone()
            idx = torch.arange(N, device=pixel_values.device)
        else:
            surprise = 1.0 - F.cosine_similarity(h, self.expectation, dim=-1)
            idx      = (surprise > self.threshold)[0].nonzero(as_tuple=True)[0]
            if len(idx) < 4:
                _, idx = torch.topk(surprise[0], 4)
            idx = torch.sort(idx)[0]
            self.expectation.mul_(1.0 - self.ema_decay).add_(h.detach() * self.ema_decay)

        self.last_sparsity = 1.0 - (len(idx) / float(N))
        hc  = h[:, idx, :]
        enc = self.model.encoder(inputs_embeds=hc)
        seq = self.model.post_layernorm(enc[0])
        return self.model.head(seq)

# ── 5. TIMING HARNESS ──────────────────────────────────────────
def iqr_timed_run(model, frames, warmup=WARMUP):
    """
    Pure GPU timing via CUDA Events — identical timing boundary to
    rigorous_timed_run() in sig1vssig2_multires.py.

    - H2D transfer isolated outside starters[i].record()
    - reset_retina() called before warmup AND before the timed run
      so both phases begin from a cold EMA state
    - Returns (latency_array_ms, embeddings, sparsities)
    """
    dummy = frames[0].to(DEVICE)
    model.reset_retina()
    for _ in range(warmup):
        model(dummy)
    model.reset_retina()
    if HAS_GPU: torch.cuda.synchronize()

    n        = len(frames)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    enders   = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if HAS_GPU else None
    lats, embs, sparsities = [], [], []

    for i, f in enumerate(frames):
        f_dev = f.to(DEVICE)                     # >>> H2D OUTSIDE TIMING <<<
        if HAS_GPU:
            starters[i].record()
            out = model(f_dev)
            enders[i].record()
        else:
            t0  = time.perf_counter()
            out = model(f_dev)
            lats.append((time.perf_counter() - t0) * 1000)
        embs.append(out.detach().cpu().float())
        sparsities.append(model.last_sparsity * 100.0)

    if HAS_GPU:
        torch.cuda.synchronize()
        lats = [s.elapsed_time(e) for s, e in zip(starters, enders)]

    return np.array(lats), embs, sparsities

# ── 6. LOAD MODEL ──────────────────────────────────────────────
print(f"Loading {MODEL_ID} (Enforcing SDPA)...")
v_full      = AutoModel.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, attn_implementation="sdpa"
).to(DEVICE).eval()
base_vision = v_full.vision_model

if os.path.exists(WEIGHTS_PATH):
    st = torch.load(WEIGHTS_PATH, map_location="cpu")
    base_vision.load_state_dict(st if not hasattr(st, "items") else st, strict=False)
    print("  Fine-tuned weights loaded.")
else:
    print("  WARNING: weights not found — using base model.")

# ── 7. SANITY CHECK ────────────────────────────────────────────
print("\nSanity check — latency must scale with resolution:")
for res in RESOLUTIONS:
    hard_gpu_reset(cooldown_sec=1)
    baked = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
    dm    = DenseSigLIP(baked)
    if HAS_GPU:
        _ = dm(make_frames(res, n=1)[0].to(DEVICE))
        torch.cuda.synchronize()
    fc   = make_frames(res, n=50)
    arr, _, _ = iqr_timed_run(dm, fc, warmup=10)
    print(f"  {res}p  N={(res // PATCH_SIZE)**2:>6}  {np.median(arr):>8.2f}ms")
    del dm, baked; gc.collect()
print()

# ── 8. MAIN BENCHMARK ──────────────────────────────────────────
results = {}

for res in RESOLUTIONS:
    N_tokens = (res // PATCH_SIZE) ** 2
    C_dim    = base_vision.config.hidden_size
    print(f"\n{'─'*70}\n  {res}p  (N={N_tokens})\n{'─'*70}")

    frames = make_frames(res)
    results[res] = {}

    print(f"\n  [ Dense Baseline ]")
    hard_gpu_reset()
    baked       = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
    dense_model = DenseSigLIP(baked)
    if HAS_GPU:
        _ = dense_model(frames[0].to(DEVICE))
        torch.cuda.synchronize()

    dense_lats, dense_embs, _ = iqr_timed_run(dense_model, frames)
    sd           = stats(dense_lats)
    dense_median = sd["median"]

    print(f"    Dense Latency: {dense_median:.3f}ms  "
          f"IQR=[{sd['p25']:.3f}, {sd['p75']:.3f}]  "
          f"P5-P95=[{sd['p05']:.3f}, {sd['p95']:.3f}]")
    results[res]["Dense"] = {
        "Method": "Dense", "Config": "—", "N_tokens": N_tokens,
        "Skip %": "0.0%", "Theor FLOP Proxy": "1.00×",
        "Emp Speedup": "1.00×", "Spd lo": "—", "Spd hi": "—",
        **sd,
    }
    del dense_model, baked; gc.collect()

    for cfg in NEUROFLOW_CONFIGS:
        print(f"\n  [ NeuroFlow: {cfg['name']} ]")
        hard_gpu_reset()
        baked    = upscale_pos_embeddings(base_vision, res).to(DEVICE, dtype=DTYPE).eval()
        nf_model = NeuroFlowSigLIP(baked, threshold=cfg["threshold"], ema_decay=cfg["ema"])
        if HAS_GPU:
            _ = nf_model(frames[0].to(DEVICE))
            torch.cuda.synchronize()

        nf_lats, nf_embs, sp = iqr_timed_run(nf_model, frames)
        s         = stats(nf_lats)
        mean_skip = float(np.mean(sp))
        fid       = float(np.mean([
            F.cosine_similarity(a, b, dim=-1).item() * 100
            for a, b in zip(nf_embs, dense_embs)
        ]))

        # Empirical speedup from distribution:
        # central: dense_median / nf_median
        # lo bound: dense_median / nf_P75 (NF is slowest)
        # hi bound: dense_median / nf_P25 (NF is fastest)
        spd    = dense_median / s["median"]
        spd_lo = dense_median / s["p75"]
        spd_hi = dense_median / s["p25"]
        theor  = get_theor_speedup(N_tokens, mean_skip, C=C_dim)

        print(f"    Skip:  {mean_skip:.1f}%   |  CosSim: {fid:.2f}%")
        print(f"    Emp Spd: {spd:.2f}×  |  Theor Spd: {theor:.2f}×  |  Lat: {s['median']:.3f}ms")
        print(f"    IQR: [{s['p25']:.3f}, {s['p75']:.3f}]ms  "
              f"P5-P95: [{s['p05']:.3f}, {s['p95']:.3f}]ms  "
              f"Spd range: [{spd_lo:.2f}×–{spd_hi:.2f}×]")

        results[res][cfg["name"]] = {
            "Method": "NeuroFlow", "Config": cfg["name"], "N_tokens": N_tokens,
            "Skip %": f"{mean_skip:.1f}%",
            "Theor FLOP Proxy": f"{theor:.2f}×",
            "Emp Speedup": f"{spd:.2f}×",
            "Spd lo": f"{spd_lo:.2f}×",
            "Spd hi": f"{spd_hi:.2f}×",
            "CosSim": f"{fid:.2f}%",
            **s,
        }
        del nf_model, baked; gc.collect()

    del frames, dense_embs; gc.collect()
    if HAS_GPU: torch.cuda.empty_cache()

# ── 9. OUTPUT ──────────────────────────────────────────────────
flattened = []
for res, configs in results.items():
    for cfg_name, data in configs.items():
        row = {"Resolution": f"{res}p", "Config": cfg_name}
        row.update(data)
        flattened.append(row)

df = pd.DataFrame(flattened)
print("\n\nFINAL IQR SUMMARY (SigLIP-2: NeuroFlow Latency Distributions)")
print("=" * 105)
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))