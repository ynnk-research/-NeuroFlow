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
# =============================================================================
"""
neuroflow_gate.py
=================
Standalone implementation of all three NeuroFlow Vision Transformer
inference architectures described in:

  "EMA-Gated Temporal Sequence Compression in Vision Transformers"

All classes wrap a HuggingFace SigLIP (or compatible) vision model and
require no fine-tuning or weight modification unless stated otherwise.

Classes
-------
NeuroFlowSiglipVisionArchA
    Late-layer MLP gating. Preserves the full O(N²) attention matrix;
    saves O(N) MLP compute for dormant tokens. Correct for O(N)-attention
    architectures (Swin, linear attention); bounded at ~1.17× wall-clock
    speedup on standard ViTs at high resolution (Amdahl ceiling).

NeuroFlowSiglipVisionArchB
    Early token elimination. Physically removes inactive tokens before the
    encoder, reducing attention to O(N_active²). Requires sparse manifold
    distillation fine-tuning to stabilise the MAP head at high sparsity.
    Achieves 55.80× wall-clock speedup at 1792p on SigLIP 2.

NeuroFlowSiglipVisionArchC  [PRIMARY CONTRIBUTION]
    Dual-Memory Reconstruction Protocol. Combines a Retinal Gate (Layer 0
    EMA, same as Architecture B) with a Cortical Cache (persistent Layer 12
    buffer). The encoder processes only active tokens; the MAP head always
    receives the full N-token K-V set reconstructed from the cache.
    Training-free. Achieves 71.55% UCF-101 zero-shot top-1 at 84.0% token
    sparsity on SigLIP base-patch16-224, retaining 92.4% of dense accuracy.

Quick Start
-----------
    from transformers import AutoModel
    from neuroflow_gate import NeuroFlowSiglipVisionArchC

    base = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    model = NeuroFlowSiglipVisionArchC(
        base.vision_model,
        threshold=0.35,   # cosine surprise threshold (MaxSparse operating point)
        ema_decay=0.01,   # EMA decay rate alpha
    )

    for frame in video_frames:
        embedding = model(frame)   # shape [1, D]

    model.reset()  # call between independent video streams

Deployment Notes
----------------
- Architecture C is constrained to resolutions ≤ 448p. Bicubic PE
  interpolation beyond 2× the native grid degrades dense accuracy
  catastrophically; this is not a NeuroFlow-specific limit.
- Call reset() before each new video stream to clear temporal state.
- The Saturation Law holds above 65% skip rate: accuracy gap variation
  (std = 0.32 pp) is statistically indistinguishable from measurement
  noise. Deploy at MaxSparse (threshold ≈ 0.35, ema_decay ≈ 0.01).
- Architecture A and B share the same EMA gating signal. Only the
  structural placement of the gate differs.

References
----------
  Degradation Law : gap ≈ 4.5 + 0.7·max(0, σ−1.0) pp  (Eq. 3 in paper)
  Saturation Law  : §5.2.2
  Capacity Law    : §5.2.4
  Amdahl Ceiling  : §5.15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE A — Late-Layer MLP Gating (Amdahl Ceiling)
# ──────────────────────────────────────────────────────────────────────────────

class NeuroFlowSiglipVisionArchA(nn.Module):
    """
    Architecture A: Late-Layer MLP Gating.

    The EMA surprise gate is inserted between the self-attention and MLP
    sub-layers of every Transformer block. Tokens whose per-patch cosine
    surprise falls below `threshold` retain their most-recently-computed
    feature vector (the keyframe anchor) rather than passing through the
    MLP. The full N-token sequence remains in memory, so self-attention
    still runs at O(N²) cost each layer.

    Wall-Clock Characteristics
    --------------------------
    At 1792p (N = 12,544), attention accounts for ~85% of frame time.
    Saving MLP compute for 82% of tokens yields at most 1.17× wall-clock
    speedup — the Amdahl ceiling. At 224p the gate overhead exceeds MLP
    savings (0.60× measured), with break-even at ~278p.

    This architecture is the correct design for O(N)-attention models
    (Swin Transformer, linear-attention variants) where MLP compute
    constitutes a larger fraction of total cost.

    Fidelity: 97.1% cosine similarity at 81.9% token sparsity.

    Parameters
    ----------
    vision_model : nn.Module
        A HuggingFace SigLIP (or compatible) vision encoder exposing
        .embeddings, .encoder.layers (iterable of Transformer blocks
        each with .self_attn and .mlp sub-modules), .post_layernorm,
        and .head.
    threshold : float
        Cosine surprise threshold τ. Tokens with surprise ≤ τ reuse the
        keyframe anchor. Default 0.15 (balanced operating point).
    ema_decay : float
        EMA decay rate α. E_t = (1−α)·E_{t−1} + α·H_t.
        Default 0.01 (slow background model).
    """

    def __init__(self, vision_model: nn.Module, threshold: float = 0.15,
                 ema_decay: float = 0.01):
        super().__init__()
        self.model = vision_model
        self.threshold = threshold
        self.ema_decay = ema_decay

        # Temporal state
        self._ema: torch.Tensor | None = None          # Layer-0 EMA  [1, N, D]
        self._keyframe: torch.Tensor | None = None     # Anchor features [1, N, D]
        self.last_sparsity: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all temporal state. Call before each new video stream."""
        self._ema = None
        self._keyframe = None

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : torch.Tensor  [1, C, H, W]
            Single preprocessed video frame. Batch size must be 1.

        reset_state : bool
            If True, clears the temporal cache before processing. Set to True 
            on the first frame of a new video stream.

        Returns
        -------
        torch.Tensor  [1, D]
            Pooled image embedding, identical in shape to the dense model.
        """

        if reset_state:
            self.reset()

        # ── 1. Patch embedding  O(N) ──────────────────────────────────
        hidden_states = self.model.embeddings(pixel_values)   # [1, N, D]
        N = hidden_states.shape[1]

        # Auto-reset if spatial resolution (sequence length) changes
        if self._ema is not None and self._ema.shape[1] != N:
            import warnings
            warnings.warn(f"NeuroFlow Arch A: Input sequence length changed from {self._ema.shape[1]} to {N}. Auto-resetting Retinal Gate and Keyframe.")
            self.reset()

        # ── 2. Compute per-patch surprise from Layer-0 EMA ────────────
        if self._ema is None or self.threshold == 0.0:
            # First frame: initialise EMA and keyframe anchor; run dense
            self._ema = hidden_states.detach().clone()
            self._keyframe = hidden_states.detach().clone()
            active_mask = torch.ones(N, dtype=torch.bool,
                                     device=pixel_values.device)
        else:
            surprise = 1.0 - F.cosine_similarity(
                hidden_states, self._ema, dim=-1)               # [1, N]
            active_mask = surprise[0] > self.threshold          # [N]  bool

            # Safety floor: always activate at least 4 tokens
            if active_mask.sum() < 4:
                _, top_idx = torch.topk(surprise[0], 4)
                active_mask = torch.zeros(N, dtype=torch.bool,
                                          device=pixel_values.device)
                active_mask[top_idx] = True

            # Update EMA unconditionally (background tracks all patches)
            self._ema.mul_(1.0 - self.ema_decay).add_(
                hidden_states.detach() * self.ema_decay)

        self.last_sparsity = 1.0 - active_mask.float().mean().item()

        # ── 3. Layer-by-layer forward with MLP gating ─────────────────
        # Architecture A keeps the full sequence in the attention pass
        # and only short-circuits the MLP for inactive tokens.
        # Inactive tokens skip the MLP entirely: their post-attention
        # residual-stream value is carried forward unchanged (zero MLP
        # contribution), which is equivalent to using the keyframe anchor
        # at that layer.  Only active tokens are routed through the MLP,
        # saving O(N_inactive) MLP FLOPs per layer.
        h = hidden_states
        for layer in self.model.encoder.layers:
            # Self-attention on ALL N tokens  O(N²)
            attn_out = layer.self_attn(h)[0]
            h = layer.layer_norm1(h + attn_out)

            # MLP: run only on active tokens to save O(N_inactive) FLOPs
            if self._keyframe is not None and (~active_mask).any():
                active_idx   = active_mask.nonzero(as_tuple=True)[0]   # [N_active]
                inactive_idx = (~active_mask).nonzero(as_tuple=True)[0] # [N_inactive]

                # Compute MLP only for active tokens
                mlp_out_active = layer.mlp(h[:, active_idx, :])        # [1, N_active, D]

                # Assemble full output: active tokens get MLP update;
                # inactive tokens carry their pre-MLP residual unchanged.
                mlp_out = torch.zeros_like(h)
                mlp_out[:, active_idx, :]   = mlp_out_active
                mlp_out[:, inactive_idx, :] = (
                    self._keyframe[:, inactive_idx, :] - h[:, inactive_idx, :]
                    # Net: h + mlp_out = keyframe for inactive tokens
                )
            else:
                # First frame or all tokens active — full MLP pass
                mlp_out = layer.mlp(h)

            h = layer.layer_norm2(h + mlp_out)

        # Update keyframe anchor to current active token representations
        self._keyframe = h.detach().clone()

        # ── 4. Pooling head ───────────────────────────────────────────
        sequence_output = self.model.post_layernorm(h)
        return self.model.head(sequence_output)


# ──────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE B — Early Token Elimination (requires fine-tuning)
# ──────────────────────────────────────────────────────────────────────────────

class NeuroFlowSiglipVisionArchB(nn.Module):
    """
    Architecture B: Early Token Elimination.

    The EMA surprise gate acts at Layer 0 (after patch embedding, before
    the first Transformer block). Inactive tokens are physically removed
    from the sequence tensor, reducing encoder attention complexity to
    O(N_active²) and yielding superlinear wall-clock speedup with skip rate.

    Wall-Clock Characteristics
    --------------------------
    SigLIP 2, GPU fp16, 1792p: 678 ms → 11.9 ms = 55.80× speedup at
    97.37% embedding fidelity after sparse manifold distillation.

    Fine-Tuning Requirement
    -----------------------
    Applied to an unmodified model, Architecture B produces MAP head
    fidelity collapse (45–76% cosine similarity) because the cross-attention
    pooling module was never trained on sparse K-V sets. Sparse manifold
    distillation fine-tunes the MAP head to pool stably from N_active ≈ 8
    tokens. See distill_siglip.py for the training procedure.

    Architecture C eliminates this requirement entirely.

    Parameters
    ----------
    vision_model : nn.Module
        Fine-tuned HuggingFace SigLIP vision encoder.
    threshold : float
        Cosine surprise threshold τ. Default 0.35 (MaxSparse).
    ema_decay : float
        EMA decay rate α. Default 0.01.
    floor_tokens : int
        Minimum active tokens forwarded to the encoder. Prevents MAP head
        attention collapse. Default 4.
    """

    def __init__(self, vision_model: nn.Module, threshold: float = 0.35,
                 ema_decay: float = 0.01, floor_tokens: int = 4):
        super().__init__()
        self.model = vision_model
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.floor_tokens = floor_tokens

        self._ema: torch.Tensor | None = None
        self.last_sparsity: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear temporal state. Call before each new video stream."""
        self._ema = None

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : torch.Tensor  [1, C, H, W]

        reset_state : bool
            If True, clears the temporal cache before processing. Set to True 
            on the first frame of a new video stream.

        Returns
        -------
        torch.Tensor  [1, D]
        """

        if reset_state:
            self.reset()

        # ── 1. Patch embedding ────────────────────────────────────────
        hidden_states = self.model.embeddings(pixel_values)   # [1, N, D]
        N = hidden_states.shape[1]

        # Auto-reset if spatial resolution (sequence length) changes
        if self._ema is not None and self._ema.shape[1] != N:
            import warnings
            warnings.warn(f"NeuroFlow Arch B: Input sequence length changed from {self._ema.shape[1]} to {N}. Auto-resetting Retinal Gate.")
            self.reset()

        # ── 2. Retinal Gate ───────────────────────────────────────────
        if self._ema is None or self.threshold == 0.0:
            self._ema = hidden_states.detach().clone()
            active_indices = torch.arange(N, device=pixel_values.device)
        else:
            surprise = 1.0 - F.cosine_similarity(
                hidden_states, self._ema, dim=-1)               # [1, N]
            active_indices = (surprise > self.threshold)[0].nonzero(
                as_tuple=True)[0]

            # Safety floor
            if len(active_indices) < self.floor_tokens:
                _, active_indices = torch.topk(
                    surprise[0], self.floor_tokens)
            active_indices = torch.sort(active_indices)[0]

            # EMA update — unconditional, tracks all patches
            self._ema.mul_(1.0 - self.ema_decay).add_(
                hidden_states.detach() * self.ema_decay)

        self.last_sparsity = 1.0 - len(active_indices) / float(N)

        # ── 3. Physical token elimination ─────────────────────────────
        # Compressed sequence: only active tokens enter the encoder.
        # Spatial ordering is preserved by the sorted index gather.
        hidden_compressed = hidden_states[:, active_indices, :]

        # ── 4. Encoder + head  O(N_active²) ──────────────────────────
        encoder_outputs = self.model.encoder(inputs_embeds=hidden_compressed)
        sequence_output = self.model.post_layernorm(encoder_outputs[0])
        return self.model.head(sequence_output)


# ──────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE C — Dual-Memory Reconstruction Protocol (training-free)
# ──────────────────────────────────────────────────────────────────────────────

class NeuroFlowSiglipVisionArchC(nn.Module):
    """
    Architecture C: Dual-Memory Reconstruction Protocol.

    The central contribution of the paper. Decouples two requirements that
    naive sparse inference conflates:

      Retinal Gate (Layer 0 EMA)
          Same cosine surprise gate as Architecture B. Determines the active
          set A^t at each frame. Active tokens are forwarded through the
          full encoder; inactive tokens are not.

      Cortical Cache (Layer 12 buffer)
          A persistent tensor C ∈ R^{N×D} storing the most recent Layer 12
          encoder output for every patch. Updated only for active patches
          each frame. Inactive patches retain their last-active Layer 12
          representation.

    At every frame the MAP head receives the full N-token sequence:
      - Active patches  → fresh Layer 12 encoder output
      - Inactive patches → cached Layer 12 output from most recent active frame

    This guarantees a structurally complete K-V set for the MAP head at all
    times, eliminating the fidelity collapse observed in Architecture B
    without fine-tuning.

    Why Identity Caching Works
    --------------------------
    Layer 12 representations of stationary patches are temporally stable:
    cos(z_i^t, z_i^{t−1}) = 0.83 (mean over inactive patches). A stale
    Layer 12 vector introduces ~34° angular error — substantially lower
    than the ≥40% error produced by any Layer 0 → Layer 12 linear
    projection (the Representation Gap). The Cortical Cache recycles the
    actual Layer 12 representation, not an approximation.

    Deployment Constraints
    ----------------------
    Constrained to resolutions ≤ 448p. Above this boundary, bicubic PE
    interpolation beyond 2× the native grid degrades the dense model
    independently of NeuroFlow. Architecture B (fine-tuned) handles
    higher resolutions.

    Empirical Laws (paper §5.2)
    ---------------------------
    Degradation Law  : gap ≈ 4.5 + 0.7·max(0, σ−1.0) pp
    Saturation Law   : gap std = 0.32 pp above 65% skip → deploy at MaxSparse
    Capacity Law     : SO400M sparse dominates base-p16-224 dense at all skips
    Spatial Displacement Law : gap predicted by subject trajectory, not speed

    Parameters
    ----------
    vision_model : nn.Module
        Unmodified HuggingFace SigLIP vision encoder. No fine-tuning needed.
        Exposes .embeddings, .encoder, .post_layernorm, .head.
    threshold : float
        Cosine surprise threshold τ. Tokens with surprise ≤ τ are routed to
        the Cortical Cache rather than the encoder.
        Recommended: 0.35 (Aggressive operating point, ~84% skip on UCF-101;
        this is the MaxSparse benchmark value reported in Table 9 of the paper).
        The Saturation Law confirms accuracy gap is invariant above τ = 0.05.
    ema_decay : float
        EMA decay rate α for the Retinal Gate.
        E_t = (1−α)·E_{t−1} + α·H_t.
        Default 0.01. Higher α = faster background model; lower α = more
        stable background representation.
    floor_tokens : int
        Minimum number of active tokens forwarded to the encoder each frame.
        Prevents degenerate single-token encoder passes. Default 4.
    """

    def __init__(self, vision_model: nn.Module, threshold: float = 0.35,
                 ema_decay: float = 0.01, floor_tokens: int = 4):
        super().__init__()
        self.model = vision_model
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.floor_tokens = floor_tokens

        # Retinal Gate state: Layer-0 EMA  [1, N, D]
        self._ema_l0: torch.Tensor | None = None

        # Cortical Cache: Layer-12 buffer  [1, N, D]
        # Stores the most recent Layer-12 encoder output per patch.
        self._cache_l12: torch.Tensor | None = None

        self.last_sparsity: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Clear all temporal state.

        Must be called between independent video streams to prevent the
        Cortical Cache from carrying stale representations across scene
        boundaries. Safe to call mid-stream to force a keyframe reset.
        """
        self._ema_l0 = None
        self._cache_l12 = None

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, reset_state: bool = False) -> torch.Tensor:
        """
        Single-frame inference pass.

        Parameters
        ----------
        pixel_values : torch.Tensor  [1, C, H, W]
            A single preprocessed video frame. Batch size must be 1.
            Normalise with SigLIP mean/std = [0.5, 0.5, 0.5].

        reset_state : bool
            If True, clears the temporal cache before processing. Set to True 
            on the first frame of a new video stream.

        Returns
        -------
        torch.Tensor  [1, D]
            Pooled image embedding identical in shape to the dense model.
            Compatible with SigLIP zero-shot classification pipelines.
        """
        if reset_state:
            self.reset()

        # ── 1. Patch embedding  O(N) ──────────────────────────────────
        h0 = self.model.embeddings(pixel_values)               # [1, N, D]
        N = h0.shape[1]
        device = pixel_values.device

        # Auto-reset if spatial resolution (sequence length) changes
        if self._ema_l0 is not None and self._ema_l0.shape[1] != N:
            import warnings
            warnings.warn(f"NeuroFlow: Input sequence length changed from {self._ema_l0.shape[1]} to {N}. Auto-resetting Cortical Cache.")
            self.reset()

        # ── 2. Retinal Gate: compute active set A^t ───────────────────
        if self._ema_l0 is None or self.threshold == 0.0:
            # Frame 1 — initialise both memories with a full dense pass
            active_indices = torch.arange(N, device=device)
            self._ema_l0 = h0.detach().clone()
        else:
            # Cosine surprise: s_i = 1 − cos(h_i^t, E_i^{t−1})
            surprise = 1.0 - F.cosine_similarity(
                h0, self._ema_l0, dim=-1)                       # [1, N]
            active_indices = (surprise > self.threshold)[0].nonzero(
                as_tuple=True)[0]                               # [N_active]

            # Safety floor: encoder must receive at least floor_tokens
            if len(active_indices) < self.floor_tokens:
                _, active_indices = torch.topk(
                    surprise[0], self.floor_tokens)
            active_indices = torch.sort(active_indices)[0]

            # EMA update — tracks ALL patches including inactive ones
            # so that the gate remains calibrated to the full background
            self._ema_l0.mul_(1.0 - self.ema_decay).add_(
                h0.detach() * self.ema_decay)

        self.last_sparsity = 1.0 - len(active_indices) / float(N)

        # ── 3. Encoder — only active tokens  O(N_active²) ────────────
        h_active = h0[:, active_indices, :]                    # [1, N_active, D]
        encoder_outputs = self.model.encoder(inputs_embeds=h_active)
        z_active = encoder_outputs[0]                          # [1, N_active, D]

        # ── 4. Cortical Cache update ──────────────────────────────────
        # Initialise cache on first frame with the dense Layer-12 output
        if self._cache_l12 is None:
            self._cache_l12 = torch.zeros(
                1, N, z_active.shape[-1], device=device,
                dtype=z_active.dtype)

        # Write fresh Layer-12 representations for active patches only
        self._cache_l12[:, active_indices, :] = z_active.detach()

        # ── 5. MAP head receives full N-token K-V set ─────────────────
        # The Cortical Cache supplies structurally complete context:
        #   - Active patches  → updated this frame (z_active)
        #   - Inactive patches → most recent Layer-12 output (cached)
        # This is the core mechanism that prevents MAP head fidelity collapse.
        full_sequence = self._cache_l12                        # [1, N, D]

        sequence_output = self.model.post_layernorm(full_sequence)
        return self.model.head(sequence_output)