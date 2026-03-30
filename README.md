# NeuroFlow: EMA-Gated Temporal Sequence Compression in Vision Transformers
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19337577.svg)](https://doi.org/10.5281/zenodo.19337577) 
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Doc License: CC BY 4.0](https://img.shields.io/badge/Doc_License-CC_BY_4.0-green.svg)](LICENSE-CC-BY.txt)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/Ynnk-Research)


> **TL;DR:** Vision Transformers waste 90% of their compute recalculating stationary asphalt. NeuroFlow tracks semantic surprise in embedding space, physically eliminating background tokens before the encoder.

<p align="center">
  <video src="https://github.com/user-attachments/assets/a34fab7f-6628-4397-9d57-5301e614d003" autoplay loop muted playsinline style="max-width:100%;">
  </video>
  <br>
  <p><em>Left: Standard SigLIP FP16 processing 100% of tokens (~722ms) @1792p. Right: NeuroFlow Architecture B eliminating 98% of stationary tokens before the encoder (~15ms) @1792p.</em></p>
</div>

This repository contains the official PyTorch implementation, verification scripts, and pre-print paper for **NeuroFlow**. 

NeuroFlow is a dynamic routing framework for Vision Transformer video inference. It exploits temporal redundancy by tracking per-patch semantic surprise via an Exponential Moving Average (EMA) of patch-level embeddings, effectively answering the architectural mismatch between $\mathcal{O}(N^2)$ self-attention and highly redundant natural video streams.

## Overview
### Key Contributions
* **Architecture C (Dual-Memory Reconstruction):** A completely *training-free* inference engine that combines a Layer 0 Retinal Gate with a Layer 12 Cortical Cache. It achieves **71.55% zero-shot top-1 accuracy at 84.0% token sparsity** on SigLIP, retaining 92.4% of dense accuracy without modifying any weights.
* **Architecture B (Extreme Wall-Clock Speedup):** Physically eliminates stationary tokens before the encoder. With sparse manifold distillation, it reduces 1792p SigLIP 2 inference from 678 ms to 11.9 ms—a **55.80× wall-clock speedup** at 97.37% embedding fidelity.
* **LLM Ablation:** Characterises the architectural boundaries of applying similarity-gated bypass to autoregressive language models (Phi-3-mini), demonstrating 0% token drift in syntactically constrained generation.

## Repository Structure
* `/core` - Production-ready PyTorch classes for NeuroFlow gating architectures (Arch A, B, and C) `neuroflow_gate.py`; distillation script `distill_siglip.py`; and llm ablation `llm_bypass_ablation.py`.
* `/scripts` - Additional Evaluation and verification scripts for testing gating architectures, reproducing benchmarks and visual outputs.
* `/paper` - LaTeX source files and PDF pre-print of the manuscript.
* `/weights` - Scripts and instructions to download the fine-tuned SigLIP v1 and SigLIP 2 models.

> **Note on Weights:** The fine-tuned 300MB model weights for Architecture B are hosted on [Hugging Face](https://huggingface.co/Ynnk-Research) and permanently archived on [Zenodo](https://doi.org/10.5281/zenodo.19337577).


Three architectures are evaluated in the paper, each representing a different placement of the gate:

| Architecture | Gate Placement | Requires Fine-Tuning | Key Result |
|---|---|---|---|
| **A** | Between attention and MLP sub-layers | No | 97.1% fidelity; bounded at 1.17× wall-clock (Amdahl ceiling) |
| **B** | Layer 0 (pre-encoder elimination) | **Yes** (sparse manifold distillation) | 55.80× wall-clock speedup at 1792p, 97.37% fidelity |
| **C** | Layer 0 + Cortical Cache (Dual-Memory) | **No** | 71.55% UCF-101 zero-shot top-1 at 84.0% token sparsity |

Architecture C is the primary contribution. It resolves the core tension in sparse ViT inference — the encoder benefits from sparsity, but the MAP pooling head requires a complete K-V set — without modifying any model weights.

## Emergent Capabilities
<br>

<p align="center">
  <video src="https://github.com/user-attachments/assets/29d789b9-c2cd-4dd9-b31e-76ebb41a199b" autoplay loop muted playsinline style="max-width:100%;">
  </video>
  <br>
  <p><em>NeuroFlow active patch clusters forwarded through the pooling head. The gate provides motion segmentation and object-level classification without a dedicated detection head or bounding-box supervision.</em></p>
</div>

## Robustness & Edge Cases
<br>

<p align="center">
  <video src="https://github.com/user-attachments/assets/4ebd0afb-251c-4247-936c-ad090cc61026" autoplay loop muted playsinline style="max-width:100%;">
  </video>
  <br>
  <p><em>Dynamic camera robustness. Even with continuous global motion, the EMA embedding gate isolates semantic surprise (edges, subjects) from structural background, sustaining 65-78% skip rates without static scenes.</em></p>
</div>

## Installation

```bash
git clone https://github.com/ynnk-research/-NeuroFlow
cd neuroflow
pip install -r requirements.txt
```

Tested on Python 3.10–3.12, PyTorch 2.x, CUDA 12. CPU-only inference is supported but not recommended above 448p.

## Quick Start

### Architecture C — Training-Free Inference (recommended starting point)

```python
from transformers import AutoModel, AutoProcessor
from neuroflow_gate import NeuroFlowSiglipVisionArchC
import torch

# Load base model — no fine-tuning needed
base = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

model = NeuroFlowSiglipVisionArchC(
    base.vision_model,
    threshold=0.35,    # MaxSparse operating point (~84% skip on surveillance footage)
    ema_decay=0.01,
).cuda().eval()

# Inference loop over video frames
for frame_pil in video_frames:
    inputs = processor(images=frame_pil, return_tensors="pt").to("cuda")
    embedding = model(inputs["pixel_values"])   # [1, 768]

# Reset between independent video streams
model.reset()
```

### Architecture B — Fine-Tuned High-Resolution Inference

```python
from transformers import AutoModel, AutoProcessor
from neuroflow_gate import NeuroFlowSiglipVisionArchB
import torch

base = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
model = NeuroFlowSiglipVisionArchB(
    base.vision_model,
    threshold=0.35,
    ema_decay=0.01,
).cuda().eval()

# Load sparse manifold distillation weights
state = torch.load("nf_archb_siglip2.pth", map_location="cuda")
model.model.load_state_dict(state, strict=False)

model.reset()
for frame_pil in video_frames:
    inputs = processor(images=frame_pil, return_tensors="pt").to("cuda")
    embedding = model(inputs["pixel_values"])
```

### Architecture A — MLP Gating (O(N)-attention target)

```python
from neuroflow_gate import NeuroFlowSiglipVisionArchA

# Architecture A is most appropriate for Swin or linear-attention models.
# On standard ViTs at high resolution it is bounded by the Amdahl ceiling.
model = NeuroFlowSiglipVisionArchA(
    base.vision_model,
    threshold=0.15,
    ema_decay=0.01,
).cuda().eval()
```

## Threshold and EMA Decay Guide

The **Saturation Law** (§6.2.2) establishes that the accuracy gap variation across skip rates is statistically indistinguishable from sampling noise above 65% skip. There is no benefit to tuning the threshold below MaxSparse once you are above this level.

| Operating Point | `threshold` | `ema_decay` | Typical Skip | Use Case |
|---|---|---|---|---|
| Conservative | 0.05 | 0.01 | ~55% | Scenes with fast camera motion |
| Balanced | 0.10 | 0.01 | ~66% | Mixed indoor/outdoor footage |
| Aggressive | 0.20 | 0.01 | ~84% | Static-background surveillance |
| MaxSparse | 0.35 | 0.01 | ~85–97% | Traffic, drone, factory footage |

The `ema_decay` parameter controls how quickly the background model adapts. Values of 0.01–0.05 work well for most scenes. Higher values (0.5–1.0) cause faster adaptation but reduce temporal stability.

## Deployment Constraints

**Architecture C** is constrained to resolutions ≤ 448p. Bicubic positional embedding interpolation beyond 2× the native grid causes catastrophic accuracy collapse in the dense base model (79% at 224p → <5% at 896p). This is not a NeuroFlow-specific limit; it applies to the base SigLIP model independently.

**Architecture B** (fine-tuned) handles any resolution but requires per-resolution distillation. Multi-resolution weights are available via the model card.

**Failure classes** for Architecture C include actions with large spatial trajectory (arc motions, jumps, traversals) and oscillating multi-object scenes. For these cases the accuracy gap is structural and cannot be recovered by threshold tuning. See §4.6 and Table 3 of the paper for the full deployment decision guide.

## Sparse Manifold Distillation (Architecture B)

To download the paper distillations from huggingface, run:
```bash
pip install huggingface_hub
python download_models.py
```

To fine-tune a model for Architecture B:

```bash
python distill_siglip.py \
    --model google/siglip2-base-patch16-224 \
    --data_dir /path/to/video/frames \
    --epochs 10 \
    --lr 5e-6 \
    --output neuroflow_edge_siglip2_finetuned.pth
```

The training objective combines L2-normalised cosine alignment with an MSE term to constrain both embedding direction and magnitude. The curriculum randomises the gate threshold and EMA decay per clip to expose the MAP head to the full range of sparsity regimes encountered at inference. See §4.3 for details.

**Warning:** Fine-tuning on a narrow label distribution induces catastrophic forgetting of general visual-semantic alignment (UCF-101 top-1: 2.40% post-fine-tuning vs 77.40% base). Use diverse training data and limit training to ≤3 epochs at LR ≤5×10⁻⁶ to mitigate this. Architecture C avoids the problem entirely.


## LLM Ablation (Phi-3-mini)

The cross-domain ablation characterises when the similarity-gating principle transfers to autoregressive language models. Wall-clock speedup is bounded at 1.000× by DRAM bandwidth on current hardware; the scientific contribution is the characterisation of the safety envelope and architectural boundary.

```bash
pip install transformers datasets tabulate
python llm_bypass_ablation.py
```

Key findings reproduced by this script: the PPL-Drift Dissociation (high token drift coexists with ΔPPl < 0.04), the Prose Cascade Attractor, and the Entropy-Bypass Correlation (code: 0% drift at 21% bypass; prose: 83% drift at any bypass rate). See §8 of the paper.


## Citation

If you use this work, please cite it as:

> Yannick Schmitt. (2026). EMA-Gated Temporal Sequence Compression in Vision Transformers. Zenodo. https://doi.org/10.5281/zenodo.19337577


## License
 * The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
 * The documentation, LaTeX source files, and PDF papers are licensed under the[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC-BY.txt).
