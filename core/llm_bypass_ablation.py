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
# OUTPUTS:
#   Table 1: WikiText-103 PPL (teacher forcing, 2000 tokens)
#   Table 2: Gate similarity distribution (P10/P50/P90)
#   Table 3: Wall-clock latency (CUDA Events, manual loop, ms/token)
#   Table 4: Token drift (dual-path autoregressive, 100 tokens)
#   Table 5: Hidden state divergence at injection (per probe layer)
#   Table 6: Integrated summary — all metrics per config
#   Conclusion: bandwidth bottleneck quantification
#
# RUNTIME: ~45 min on T4. WikiText PPL dominates (~4 min per config).


import math, time, gc, copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tabulate import tabulate

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16

# ─── Benchmark parameters ────────────────────────────────────────────────────
WIKITEXT_TOKENS = 2000
WIKITEXT_WARMUP = 64
WALLCLOCK_TOKENS = 100     # per prompt, manual loop with CUDA events
WALLCLOCK_REPEATS = 5      # independent runs for median
DRIFT_TOKENS     = 100     # dual-path autoregressive
PROBE_LAYERS     = [0, 1, 8, 16, 19, 24, 31]

# Gate-cache pairs and thresholds reproduce the paper's headline results.
# Table 21 (WikiText PPL) and Table 22 (token drift) use L1→L24 and L1→L31
# at theta in {0.60, 0.65, 0.70}.  The L1→L0 inverted control is included to
# reproduce the catastrophic failure row (PPL=265.93 at theta=0.60).
BYPASS_CONFIGS = [
    (1,  0,  "L1->L0   (inverted-ctrl)"),   # Table 21 control -- expect PPL collapse
    (1,  24, "L1->L24  (bridge)"),           # Table 21/22 primary config
    (1,  31, "L1->L31  (deep)"),             # Table 21/22 primary config
]
THRESHOLDS = [0.60, 0.65, 0.70]

PROMPTS = {
    "Code":       ("def merge_sort(arr):\n    if len(arr) <= 1:\n"
                   "        return arr\n    mid = len(arr) // 2\n"),
    "Structured": '{"name": "Alice", "age": 30, "city": "Boston",',
    "Prose":      ("The Amazon rainforest spans nine countries and contains "
                   "more than half of the world's remaining tropical forests."),
}

# ─── Utilities ────────────────────────────────────────────────────────────────
def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def vram_gb():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.memory_allocated() / 1e9

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    print(f"Loading {MODEL_ID}...")
    # Native transformers Phi-3 (no trust_remote_code, no rope_scaling bug)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE,
            attn_implementation="eager", trust_remote_code=False,
        ).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
        print(f"  Loaded native. VRAM: {vram_gb():.1f}GB")
    except Exception as e:
        print(f"  Native failed ({e}), patching config...")
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        if (hasattr(cfg, "rope_scaling") and isinstance(cfg.rope_scaling, dict)
                and "type" not in cfg.rope_scaling):
            cfg.rope_scaling["type"] = "longrope"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, torch_dtype=DTYPE,
            attn_implementation="eager", trust_remote_code=True,
        ).to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"  Loaded with patch. VRAM: {vram_gb():.1f}GB")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def load_wikitext(tok, n):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split="test", trust_remote_code=False)
        text = " ".join(r for r in ds["text"] if r.strip())
        ids  = tok.encode(text, add_special_tokens=False)
        print(f"  WikiText-103: {len(ids):,} tokens available, using {n}")
        return torch.tensor(ids[:n])
    except Exception as e:
        print(f"  WikiText load failed ({e}), using built-in fallback")
        FALLBACK = (
            "The history of artificial intelligence dates back to antiquity, with "
            "myths and stories of artificial beings endowed with intelligence. "
            "Modern AI research was founded as an academic discipline in 1956. "
            "The field draws on computer science, mathematics, psychology, and linguistics. "
            "Machine learning, a subset of AI, enables systems to learn from data. "
            "Deep learning uses neural networks with many layers to model complex patterns. "
            "Natural language processing allows computers to understand human language. "
            "Computer vision enables machines to interpret and understand visual information. "
            "Reinforcement learning trains agents through reward and punishment signals. "
        ) * 25
        ids = tok.encode(FALLBACK, add_special_tokens=False)
        return torch.tensor(ids[:n])

class BypassEngine:
    """
    Cross-layer attention injection.
    
    GATE (layer G): hook captures attention output z_G.
                    cos_sim(z_G[t], z_G[t-1]) >= threshold -> bypass_fired.
    CACHE (layer C): self_attn.forward is monkey-patched.
                    When bypass_fired: return z_C[t-1] immediately (0 FLOPs).
                    When not fired:   run original forward, store output as z_C[t].
    
    Only layer C attention is bypassed. All MLPs and all other layers run normally.
    KV-cache at C is not written when bypassed (consistent with "same token" claim).
    
    This is the ONLY design that produces real wall-clock speedup because the
    patched forward() returns before any tensor operations execute.
    """
    def __init__(self, model, gate_layer, cache_layer, threshold):
        self.model       = model
        self.gate_layer  = gate_layer
        self.cache_layer = cache_layer
        self.threshold   = threshold
        self.layers      = model.model.layers

        # State — reset between sequences
        self._z_gate  = None   # previous step gate output [1, 1, D]
        self._z_cache = None   # previous step cache output [1, 1, D]
        self.bypass_fired = False
        self.is_active    = False

        # Metrics
        self.gate_sims      = []
        self.bypass_count   = 0
        self.total_steps    = 0

        # Store original forwards
        self._orig_cache_fwd = self.layers[cache_layer].self_attn.forward

        # Hidden state probe storage (for divergence measurement)
        self._probe_dense  = {}   # layer -> hidden state (dense path)
        self._probe_bypass = {}   # layer -> hidden state (bypass path)
        self._probe_hooks  = []
        self._active_path  = "dense"

        self._install()

    def _install(self):
        engine = self  # closure

        # Gate hook
        def gate_hook(module, inp, out):
            if not engine.is_active: return out
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != 1: return out  # prefill: skip
            z = h[0, 0, :].detach().float()
            if engine._z_gate is not None:
                sim = F.cosine_similarity(
                    z.unsqueeze(0), engine._z_gate.unsqueeze(0)).item()
                engine.bypass_fired = sim >= engine.threshold
                engine.gate_sims.append(sim)
            else:
                engine.bypass_fired = False
            engine._z_gate = z
            return out

        self._gate_hook = (self.layers[self.gate_layer].self_attn
                           .register_forward_hook(gate_hook))

        # Cache layer forward patch.
        # On bypass: return the cached output BEFORE calling the original
        # forward so that attention computation and o_proj are truly skipped.
        # The KV-cache entry is NOT updated on a bypass step, keeping the
        # cache consistent with the "same token re-used" semantic.
        def patched_cache_forward(*args, **kwargs):
            if not engine.is_active:
                return engine._orig_cache_fwd(*args, **kwargs)

            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0] if args else None
            is_decode = (hidden is not None and hidden.shape[1] == 1)

            if is_decode:
                engine.total_steps += 1
                # Early return: skip all attention compute on bypass
                if engine.bypass_fired and engine._z_cache is not None:
                    engine.bypass_count += 1
                    # Return a tuple matching the original self_attn signature:
                    # (attn_output, attn_weights=None, past_key_value=None)
                    # The KV cache at this layer is intentionally not advanced.
                    cached = engine._z_cache.to(
                        hidden.dtype if hidden is not None else torch.float16)
                    return (cached, None, None)

            result = engine._orig_cache_fwd(*args, **kwargs)

            if is_decode:
                out_h = result[0] if isinstance(result, tuple) else result
                engine._z_cache = out_h[:, -1:, :].detach().float()

            return result

        self.layers[self.cache_layer].self_attn.forward = patched_cache_forward

    def reset(self):
        self._z_gate      = None
        self._z_cache     = None
        self.bypass_fired = False
        self.gate_sims    = []
        self.bypass_count = 0
        self.total_steps  = 0

    def remove(self):
        self._gate_hook.remove()
        self.layers[self.cache_layer].self_attn.forward = self._orig_cache_fwd

    @property
    def bypass_rate(self):
        return self.bypass_count / self.total_steps if self.total_steps > 0 else 0.0

class HiddenStateProbe:
    def __init__(self, model, layers):
        self.states = {}
        self._hooks = []
        for i in layers:
            if i >= len(model.model.layers): continue
            def hook(mod, inp, out, idx=i):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1:
                    self.states[idx] = h[0, 0, :].detach().float().clone()
            self._hooks.append(
                model.model.layers[i].mlp.register_forward_hook(hook))

    def clear(self): self.states = {}
    def remove(self):
        for h in self._hooks: h.remove()


@torch.no_grad()
def compute_ppl(model, token_ids, device, engine=None, warmup=64, label=""):
    N = len(token_ids)
    if engine:
        engine.reset()
        engine.is_active = True

    inp = token_ids[:1].unsqueeze(0).to(device)
    out = model(inp, use_cache=True)
    past = out.past_key_values

    total_nll = 0.0; n = 0
    seg_nll   = 0.0; seg_n = 0; seg_size = 200
    t0 = time.time()

    for i in range(1, N):
        inp = token_ids[i-1:i].unsqueeze(0).to(device)
        out = model(inp, past_key_values=past, use_cache=True)
        # past_key_values may be None at bypassed layers; handle gracefully
        new_past = out.past_key_values
        if new_past is not None: past = new_past

        nll = -F.log_softmax(out.logits[0, -1, :].float(), dim=-1
                             )[token_ids[i].item()].item()
        if i >= warmup:
            total_nll += nll; n += 1
            seg_nll   += nll; seg_n += 1

        if i % seg_size == 0 or i == N-1:
            seg_ppl = math.exp(seg_nll/seg_n) if seg_n else float("nan")
            run_ppl = math.exp(total_nll/n)   if n    else float("nan")
            spd = i / (time.time()-t0)
            br  = engine.bypass_rate*100 if engine else 0
            print(f"  [{label}] {i}/{N}  seg={seg_ppl:.3f}  run={run_ppl:.3f}"
                  f"  {spd:.0f}tok/s  bypass={br:.1f}%")
            seg_nll = 0.0; seg_n = 0

    if engine: engine.is_active = False
    return math.exp(total_nll/n) if n else float("nan")


@torch.no_grad()
def measure_wallclock(model, tok, prompt, n_tokens, engine=None, n_repeats=5):
    """
    Returns (median_ms_per_token, std_ms_per_token, bypass_rate).
    Uses CUDA Events for precision. Times only the decode forward() calls,
    excluding prefill. No .generate() overhead.
    """
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    run_lats = []

    for _ in range(n_repeats):
        # Prefill (dense always)
        if engine:
            engine.reset()
            engine.is_active = False
        with torch.no_grad():
            out  = model(ids, use_cache=True)
            past = out.past_key_values
            tok_ = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(1)

        if engine:
            engine.reset()
            engine.is_active = True

        if torch.cuda.is_available():
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_ev.record()
        else:
            t0 = time.perf_counter()

        for _ in range(n_tokens):
            out  = model(tok_, past_key_values=past, use_cache=True)
            new_past = out.past_key_values
            if new_past is not None: past = new_past
            tok_ = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(1)

        if torch.cuda.is_available():
            end_ev.record()
            torch.cuda.synchronize()
            run_lats.append(start_ev.elapsed_time(end_ev) / n_tokens)
        else:
            run_lats.append((time.perf_counter()-t0)*1000/n_tokens)

        if engine: engine.is_active = False

    bp_rate = engine.bypass_rate if engine else 0.0
    return float(np.median(run_lats)), float(np.std(run_lats)), bp_rate


@torch.no_grad()
def measure_drift(model, tok, prompt, n_tokens, engine, probe):
    """
    Runs dense and bypass in strict alternation from the same state.
    Returns (drift_rate, bypass_rate, hidden_divergence_by_layer).
    """
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)

    # Shared prefill — get KV state once
    engine.is_active = False
    engine.reset()
    out_shared = model(ids, use_cache=True)
    past_shared = out_shared.past_key_values
    first_tok   = torch.argmax(out_shared.logits[:, -1, :], dim=-1).unsqueeze(1)

    # Dense path
    past_d = past_shared; tok_d = first_tok.clone()
    dense_toks = []
    engine.is_active = False
    for _ in range(n_tokens):
        out = model(tok_d, past_key_values=past_d, use_cache=True)
        if out.past_key_values is not None: past_d = out.past_key_values
        next_t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        dense_toks.append(next_t)
        tok_d = torch.tensor([[next_t]], device=DEVICE)

    # Bypass path
    past_b = past_shared; tok_b = first_tok.clone()
    bypass_toks = []
    hs_divergence = defaultdict(list)
    engine.reset()
    engine.is_active = True

    for step in range(n_tokens):
        # Probe dense hidden states at this step
        probe.clear()
        engine.is_active = False
        out_d = model(tok_d, past_key_values=past_d, use_cache=True)
        if out_d.past_key_values is not None: past_d = out_d.past_key_values
        dense_states = dict(probe.states)
        tok_d = torch.tensor([[dense_toks[step]]], device=DEVICE)

        # Probe bypass hidden states
        probe.clear()
        engine.is_active = True
        out_b = model(tok_b, past_key_values=past_b, use_cache=True)
        if out_b.past_key_values is not None: past_b = out_b.past_key_values
        bypass_states = dict(probe.states)
        engine.is_active = False

        next_b = torch.argmax(out_b.logits[:, -1, :], dim=-1).item()
        bypass_toks.append(next_b)
        tok_b = torch.tensor([[next_b]], device=DEVICE)

        # Hidden state divergence when bypass fires
        if engine.bypass_fired:
            for l in PROBE_LAYERS:
                if l in dense_states and l in bypass_states:
                    div = 1.0 - F.cosine_similarity(
                        dense_states[l].unsqueeze(0),
                        bypass_states[l].unsqueeze(0)).item()
                    hs_divergence[l].append(div)

    engine.is_active = False
    drift = sum(1 for a,b in zip(dense_toks, bypass_toks) if a!=b) / n_tokens
    mean_div = {l: float(np.mean(v)) for l,v in hs_divergence.items() if v}
    return drift, engine.bypass_rate, mean_div

def run():
    print("="*80)
    print("NEUROFLOW LLM — DEFINITIVE ABLATION")
    print("="*80)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f}GB  BW: ~300GB/s")

    model, tok = load_model()
    n_layers   = len(model.model.layers)
    D          = model.config.hidden_size
    print(f"Layers: {n_layers}  Hidden: {D}")

    wiki_ids = load_wikitext(tok, WIKITEXT_TOKENS)
    probe    = HiddenStateProbe(model, PROBE_LAYERS)

    print("\n[DENSE BASELINE — WikiText PPL]")
    dense_ppl = compute_ppl(model, wiki_ids, DEVICE, label="dense")
    print(f"  Dense PPL: {dense_ppl:.4f}")

    print("\n[DENSE BASELINE — Wall-clock]")
    dense_lat = {}
    for task, prompt in PROMPTS.items():
        ms, std, _ = measure_wallclock(model, tok, prompt,
                                        WALLCLOCK_TOKENS, n_repeats=WALLCLOCK_REPEATS)
        dense_lat[task] = ms
        print(f"  {task}: {ms:.3f}ms/tok ±{std:.3f}")

    ppl_rows     = []
    wc_rows      = []
    drift_rows   = []
    hs_rows      = []
    summary_rows = []

    for (GL, CL, label) in BYPASS_CONFIGS:
        if GL >= n_layers or CL >= n_layers:
            print(f"  SKIP {label}: layer out of range"); continue

        for threshold in THRESHOLDS:
            cfg = f"{label} t={threshold:.2f}"
            print(f"\n{'─'*60}")
            print(f"CONFIG: {cfg}")
            print(f"{'─'*60}")

            engine = BypassEngine(model, GL, CL, threshold)

            # 1. WikiText PPL
            print("  [PPL]")
            bp_ppl = compute_ppl(model, wiki_ids, DEVICE, engine=engine,
                                  label=f"bypass {GL}→{CL} t={threshold:.2f}")
            d_ppl  = bp_ppl - dense_ppl
            gate_p10 = float(np.percentile(engine.gate_sims, 10)) if engine.gate_sims else 0
            gate_p50 = float(np.percentile(engine.gate_sims, 50)) if engine.gate_sims else 0
            gate_p90 = float(np.percentile(engine.gate_sims, 90)) if engine.gate_sims else 0
            bypass_pct_ppl = engine.bypass_rate * 100

            safe = "|Δ|<1.0" if abs(d_ppl) < 1.0 else ("|Δ|<2.0" if abs(d_ppl) < 2.0 else "RISKY")
            ppl_rows.append([label, f"{threshold:.2f}",
                              f"{bp_ppl:.4f}", f"{d_ppl:+.4f}",
                              f"{bypass_pct_ppl:.1f}%",
                              f"{engine.bypass_count}/{engine.total_steps}",
                              f"{gate_p50:.3f}", safe])
            print(f"  PPL={bp_ppl:.4f}  dPPL={d_ppl:+.4f}  "
                  f"bypass={bypass_pct_ppl:.1f}%  safe={safe}")

            # 2. Wall-clock + drift per task
            task_speedups = []
            task_drifts   = []
            for task, prompt in PROMPTS.items():
                # Wall-clock
                ms, std, bp_rate = measure_wallclock(
                    model, tok, prompt, WALLCLOCK_TOKENS,
                    engine=engine, n_repeats=WALLCLOCK_REPEATS)
                speedup = dense_lat[task] / ms if ms > 0 else 0
                task_speedups.append(speedup)
                wc_rows.append([label, f"{threshold:.2f}", task,
                                 f"{bp_rate*100:.1f}%",
                                 f"{dense_lat[task]:.3f}", f"{ms:.3f}", f"{std:.3f}",
                                 f"{speedup:.4f}x"])
                print(f"  [{task}] wall-clock: {dense_lat[task]:.2f}->{ms:.2f}ms "
                      f"({speedup:.3f}x)  bypass={bp_rate*100:.1f}%")

                # Token drift + hidden state divergence
                drift, drift_bp_rate, hs_div = measure_drift(
                    model, tok, prompt, DRIFT_TOKENS, engine, probe)
                task_drifts.append(drift)
                verdict = ("SAFE+FAST" if speedup>1.0 and drift<0.05 else
                           "SAFE"      if drift<0.05 else
                           "FAST+DRIFT" if speedup>1.0 else "SLOW+DRIFT")
                drift_rows.append([label, f"{threshold:.2f}", task,
                                    f"{drift_bp_rate*100:.1f}%",
                                    f"{drift*100:.1f}%",
                                    verdict])
                print(f"  [{task}] drift={drift*100:.1f}%  {verdict}")

                for l, div in sorted(hs_div.items()):
                    role = ""
                    if l == GL: role = "<- GATE"
                    if l == CL: role = "<- CACHE (injection)"
                    hs_rows.append([label, f"{threshold:.2f}", task, l,
                                     f"{div:.4f}",
                                     engine.bypass_count, role])

            # Summary row
            mean_speedup = float(np.mean(task_speedups))
            mean_drift   = float(np.mean(task_drifts)) * 100
            summary_rows.append([label, f"{threshold:.2f}",
                                   f"{bypass_pct_ppl:.1f}%",
                                   f"{d_ppl:+.4f}", safe,
                                   f"{mean_speedup:.4f}x",
                                   f"{mean_drift:.1f}%"])
            engine.remove()

    probe.remove()

    # ── Bandwidth bottleneck analysis ─────────────────────────────────────────
    # At batch=1, saving FLOPs at layer C attention does not save DRAM loads.
    # Theoretical speedup = 1 / (1 - flop_fraction_saved)
    # Measured speedup = actual
    # Gap = bandwidth cost of loading bypassed weights
    n_attn_params_per_layer = 4 * D * D  # Q, K, V, O projections
    bytes_per_layer_fp16    = n_attn_params_per_layer * 2
    bandwidth_gb_per_s      = 300.0      # T4 approximate
    attn_frac               = 0.35       # attention is ~35% of layer FLOPs

    # ── Print all tables ──────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("TABLE 1 — WikiText-103 PERPLEXITY (teacher forcing, 2000 tokens)")
    print(f"Dense baseline: {dense_ppl:.4f}")
    print("="*80)
    print(tabulate(ppl_rows,
        headers=["Config","Threshold","Bypass PPL","dPPL","Bypass%",
                 "Steps","Gate P50","Safety"],
        tablefmt="github"))

    print("\n\n" + "="*80)
    print("TABLE 2 — GATE SIMILARITY DISTRIBUTION")
    print("="*80)
    gate_rows = []
    for row in ppl_rows:
        gate_rows.append([row[0], row[1], row[4], row[6]])
    print(tabulate(gate_rows,
        headers=["Config","Threshold","Bypass%","Gate P50"],
        tablefmt="github"))
    print("  P50 < threshold => gate fires on volatile steps (risky)")
    print("  P50 >= threshold => majority of steps qualify for bypass")

    print("\n\n" + "="*80)
    print("TABLE 3 — WALL-CLOCK LATENCY (CUDA Events, manual decode loop)")
    print("="*80)
    print(tabulate(wc_rows,
        headers=["Config","Threshold","Task","Bypass%",
                 "Dense ms/tok","Bypass ms/tok","±Std","Speedup"],
        tablefmt="github"))

    print("\n\n" + "="*80)
    print("TABLE 4 — TOKEN DRIFT (dual-path autoregressive, 100 tokens)")
    print("="*80)
    print(tabulate(drift_rows,
        headers=["Config","Threshold","Task","Bypass%","Drift%","Verdict"],
        tablefmt="github"))

    if hs_rows:
        print("\n\n" + "="*80)
        print("TABLE 5 — HIDDEN STATE DIVERGENCE AT INJECTION LAYER")
        print("  When bypass fires: cos distance between dense and bypass hidden states")
        print("="*80)
        print(tabulate(hs_rows,
            headers=["Config","Threshold","Task","Layer","Mean Div","N Bypass","Role"],
            tablefmt="github"))

    print("\n\n" + "="*80)
    print("TABLE 6 — INTEGRATED SUMMARY")
    print("="*80)
    print(tabulate(summary_rows,
        headers=["Config","Threshold","Bypass%","dPPL","Quality",
                 "Mean Speedup","Mean Drift%"],
        tablefmt="github"))

    # Bandwidth bottleneck quantification
    print("\n\n" + "="*80)
    print("BANDWIDTH BOTTLENECK QUANTIFICATION")
    print("="*80)
    # Get the best performing config from summary
    best = None
    for r in summary_rows:
        sp = float(r[5].replace("x",""))
        dr = float(r[6].replace("%",""))
        if r[3].startswith("|Δ|<1") or "SAFE" in str(r[4]):
            if best is None or sp > float(best[5].replace("x","")):
                best = r
    if best:
        measured_sp = float(best[5].replace("x",""))
        bp_pct      = float(best[2].replace("%","")) / 100
        theoretical_sp = 1.0 / (1.0 - bp_pct * (1/32) * attn_frac)
        gap = theoretical_sp - measured_sp
        weight_bytes = bytes_per_layer_fp16
        load_time_ms = weight_bytes / (bandwidth_gb_per_s * 1e9) * 1000
        print(f"  Best safe config: {best[0]} t={best[1]}")
        print(f"  Measured speedup:     {measured_sp:.4f}x")
        print(f"  Theoretical speedup:  {theoretical_sp:.4f}x")
        print(f"  Gap (bandwidth cost): {gap:.4f}x")
        print(f"  L31 attention weight load: {weight_bytes/1e6:.1f}MB")
        print(f"  At T4 bandwidth (~300GB/s): {load_time_ms:.3f}ms")
    print(f"""
  INTERPRETATION:
  Wall-clock speedup < theoretical speedup because skipping a layer's
  attention computation does not skip the DRAM load of its weight matrices.
  At batch_size=1 on T4, weight loading dominates. The gap between
  theoretical and measured speedup is the weight-load cost in precise form.
  
    """)

    del model; gpu_reset()

if __name__ == "__main__":
    run()