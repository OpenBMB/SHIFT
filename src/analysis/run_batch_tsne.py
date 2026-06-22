#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch t-SNE analysis for models — gate conflict detection proof.

For each model:
  1. Extract layer-dim gate vectors for all test samples
  2. t-SNE visualization (con vs mem)
  3. Violin plot per layer
  4. Classification accuracy / F1 / AUC

Results saved to paper_figures/t-SNE/<model_name>/

Usage:
  python run_batch_tsne.py --cuda 0
  python run_batch_tsne.py --cuda 0 --model Llama3.2-1B   # run single model
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ── Root paths ───────────────────────────────────────────────────────────────

ROOT = Path("/path/to/your/proj/root") # this refers to the parent dir to your project
TEST_DATA_DIR = ROOT / "datasets" / "test"
OUTPUT_ROOT = Path(__file__).parent

MAX_INPUT_LENGTH = 7000
MAX_SAMPLES_PER_DATASET = 200

# ── Model registry ──────────────────────────────────────────────────────────
# ⚠️: You should change the following dirs to yours

MODELS = {
    "Llama3.1-8B": {
        "arch": "llama",
        "ckpt": ROOT / "ffn/scratch/Llama3.1-8B-Instruct/scalar-gate/gate_grpo_all_layers/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Llama3.1-8B-Instruct",
        "base": ROOT / "models/ffn/Llama3.1-8B-Instruct",
        "tokenizer": ROOT / "models/ffn/Llama3.1-8B-Instruct",
        "num_layers": 32,
        "enable_thinking": False,
    },
    "Llama3.2-1B": {
        "arch": "llama",
        "ckpt": ROOT / "ffn/scratch/Llama3.2-1B-Instruct/scalar-gate/gate_grpo_all_layers/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Llama3.2-1B-Instruct",
        "base": ROOT / "models/ffn/Llama3.2-1B-Instruct",
        "tokenizer": ROOT / "models/ffn/Llama3.2-1B-Instruct",
        "num_layers": 16,
        "enable_thinking": False,
    },
    "Llama3.2-3B": {
        "arch": "llama",
        "ckpt": ROOT / "ffn/scratch/Llama3.2-3B-Instruct/scalar-gate/gate_grpo_all_layers/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Llama3.2-3B-Instruct",
        "base": ROOT / "models/ffn/Llama3.2-3B-Instruct",
        "tokenizer": ROOT / "models/ffn/Llama3.2-3B-Instruct",
        "num_layers": 28,
        "enable_thinking": False,
    },
    "Qwen3-0.6B": {
        "arch": "qwen3",
        "ckpt": ROOT / "ffn/scratch/Qwen3-0.6B/scalar-gate/gate_grpo_all_layers_nothinking/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Qwen3-0.6B",
        "base": ROOT / "models/ffn/Qwen3-0.6B",
        "tokenizer": ROOT / "models/ffn/Qwen3-0.6B",
        "num_layers": 28,
        "enable_thinking": False,
    },
    "Qwen3-1.7B": {
        "arch": "qwen3",
        "ckpt": ROOT / "ffn/scratch/Qwen3-1.7B/scalar-gate/gate_grpo_all_layers_nothinking/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Qwen3-1.7B",
        "base": ROOT / "models/ffn/Qwen3-1.7B",
        "tokenizer": ROOT / "models/ffn/Qwen3-1.7B",
        "num_layers": 28,
        "enable_thinking": False,
    },
    "Qwen3-8B": {
        "arch": "qwen3",
        "ckpt": ROOT / "ffn/scratch/Qwen3-8B/scalar-gate/gate_grpo_all_layers/checkpoints/ckpt_000150/hf_model",
        "code": ROOT / "models/ffn/Qwen3-8B",
        "base": ROOT / "models/ffn/Qwen3-8B",
        "tokenizer": ROOT / "models/ffn/Qwen3-8B",
        "num_layers": 36,
        "enable_thinking": False,
    },
}

# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions based on the provided background information."
    "The background information may be incorrect, so you should judge whether to believe yourself or to believe the background information."
    "You should think about the reasoning process and then provide the answer based on the given context."
)

PROMPT_TEMPLATE = (
    "Background:\n{context}\n\n"
    "Task Instruction:\n"
    "Answer the question with the given background information above. "
    "Provide your final answer in <answer> </answer> tags, "
    "for example <answer>Petri Alanko</answer>.\n\n"
    "Q: {question}\n\n"
    "A: "
)

# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path, code_dir, arch, device="cuda:0"):
    code_dir = str(code_dir)
    if arch == "llama":
        cfg_mod, model_mod = "configuration_llama", "modeling_llama"
        cfg_cls, model_cls = "LlamaConfig", "LlamaForCausalLM"
    else:
        cfg_mod, model_mod = "configuration_qwen3", "modeling_qwen3"
        cfg_cls, model_cls = "Qwen3Config", "Qwen3ForCausalLM"

    for m in [cfg_mod, model_mod]:
        sys.modules.pop(m, None)
    if code_dir in sys.path:
        sys.path.remove(code_dir)
    sys.path.insert(0, code_dir)

    config_module = __import__(cfg_mod)
    model_module = __import__(model_mod)
    Config = getattr(config_module, cfg_cls)
    ModelCls = getattr(model_module, model_cls)

    config = Config.from_pretrained(model_path)
    config._attn_implementation = "eager"
    # Normalize pad_token_id: must be int or None, not a list
    pad = getattr(config, "pad_token_id", None)
    if isinstance(pad, list):
        config.pad_token_id = pad[0] if pad else None
    if config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id if not isinstance(config.eos_token_id, list) else config.eos_token_id[0]

    model = ModelCls(config)

    from safetensors.torch import load_file
    st_file = Path(model_path) / "model.safetensors"
    idx_file = Path(model_path) / "model.safetensors.index.json"
    if st_file.exists():
        state_dict = load_file(str(st_file))
    elif idx_file.exists():
        with open(idx_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for i, fname in enumerate(shard_files):
            print(f"    Loading shard {i+1}/{len(shard_files)}: {fname}")
            state_dict.update(load_file(str(Path(model_path) / fname)))
    else:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict
    if missing:
        print(f"  [warn] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    return model


def load_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_test_data(tokenizer, enable_thinking=False, max_samples=MAX_SAMPLES_PER_DATASET):
    all_samples = []
    test_files = sorted(TEST_DATA_DIR.glob("*.jsonl"))
    for f in test_files:
        ds_name = f.stem
        samples = []
        with open(f) as fp:
            for line in fp:
                item = json.loads(line)
                is_par_right = item.get("is_parametric_answer_right", False)
                context = (
                    (item.get("cf_context", "") or item["context"])
                    if is_par_right else item["context"]
                )
                user_content = PROMPT_TEMPLATE.format(
                    context=context, question=item["question"]
                )
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ]
                kwargs = dict(tokenize=True, add_generation_prompt=True)
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        input_ids = tokenizer.apply_chat_template(
                            messages, enable_thinking=enable_thinking, **kwargs
                        )
                    except TypeError:
                        input_ids = tokenizer.apply_chat_template(messages, **kwargs)
                else:
                    input_ids = tokenizer.apply_chat_template(messages, **kwargs)

                if len(input_ids) > MAX_INPUT_LENGTH:
                    continue
                samples.append({
                    "input_ids": input_ids,
                    "group": "con" if not is_par_right else "mem",
                    "dataset": ds_name,
                })
                if len(samples) >= max_samples:
                    break
        all_samples.extend(samples)
        print(f"    {ds_name}: {len(samples)} (con={sum(1 for s in samples if s['group']=='con')}, "
              f"mem={sum(1 for s in samples if s['group']=='mem')})")
    return all_samples


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_gates(model, samples, num_layers):
    gate_vectors = np.zeros((len(samples), num_layers), dtype=np.float32)

    for idx, sample in enumerate(samples):
        if idx % 100 == 0:
            print(f"      Forward {idx}/{len(samples)} ...")

        input_ids = torch.tensor([sample["input_ids"]], device=model.device)
        last_pos = input_ids.shape[1] - 1

        gate_cache = {}
        hooks = []

        for li in range(num_layers):
            layer = model.model.layers[li]
            if hasattr(layer, "ffn_gate"):
                def make_hook(layer_idx):
                    def hook_fn(module, inp, out):
                        g = 2.0 * torch.sigmoid(out)
                        g = g.clamp(min=0.7, max=1.3)
                        gate_cache[layer_idx] = g[0, last_pos, 0].item()
                    return hook_fn
                hooks.append(layer.ffn_gate.register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        for li, val in gate_cache.items():
            gate_vectors[idx, li] = val

        del input_ids
        if idx % 200 == 0:
            torch.cuda.empty_cache()

    return gate_vectors


# ── Visualization ────────────────────────────────────────────────────────────

def plot_tsne(trained_gates, base_gates, labels, out_dir, model_name):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    con_mask = np.array(labels) == "con"
    mem_mask = np.array(labels) == "mem"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trained
    ax = axes[0]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb = tsne.fit_transform(trained_gates)
    ax.scatter(emb[con_mask, 0], emb[con_mask, 1], c="#e74c3c", alpha=0.5, s=12, label="Conflict", edgecolors="none")
    ax.scatter(emb[mem_mask, 0], emb[mem_mask, 1], c="#3498db", alpha=0.5, s=12, label="Non-conflict", edgecolors="none")
    ax.set_title(f"Trained ({model_name})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])

    # Base
    ax = axes[1]
    base_var = np.var(base_gates) if base_gates is not None else 0
    if base_gates is not None and base_var > 1e-10:
        tsne_b = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        emb_b = tsne_b.fit_transform(base_gates)
        ax.scatter(emb_b[con_mask, 0], emb_b[con_mask, 1], c="#e74c3c", alpha=0.5, s=12, label="Conflict", edgecolors="none")
        ax.scatter(emb_b[mem_mask, 0], emb_b[mem_mask, 1], c="#3498db", alpha=0.5, s=12, label="Non-conflict", edgecolors="none")
    else:
        gate_val = base_gates[0, 0] if base_gates is not None else 1.0
        ax.text(0.5, 0.5, f"All gates = {gate_val:.4f}\n(no learned signal)",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    ax.set_title(f"Base ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"t-SNE of Gate Activations — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(out_dir / f"gate_tsne{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violin(trained_gates, labels, out_dir, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    num_layers = trained_gates.shape[1]
    con_mask = np.array(labels) == "con"
    mem_mask = np.array(labels) == "mem"

    fig, ax = plt.subplots(figsize=(max(14, num_layers * 0.6), 6))
    pos_c = np.arange(num_layers) * 3
    pos_m = np.arange(num_layers) * 3 + 1

    con_data = [trained_gates[con_mask, l] for l in range(num_layers)]
    mem_data = [trained_gates[mem_mask, l] for l in range(num_layers)]

    vp_c = ax.violinplot(con_data, positions=pos_c, showmeans=True, widths=0.8)
    vp_m = ax.violinplot(mem_data, positions=pos_m, showmeans=True, widths=0.8)

    for body in vp_c["bodies"]:
        body.set_facecolor("#e74c3c"); body.set_alpha(0.6)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp_c: vp_c[part].set_color("#c0392b")
    for body in vp_m["bodies"]:
        body.set_facecolor("#3498db"); body.set_alpha(0.6)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp_m: vp_m[part].set_color("#2980b9")

    ax.set_xticks(np.arange(num_layers) * 3 + 0.5)
    ax.set_xticklabels([str(i) for i in range(num_layers)], fontsize=7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Gate Value")
    ax.set_title(f"Gate Distribution: Con vs Mem — {model_name}", fontsize=13, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=[Patch(facecolor="#e74c3c", alpha=0.6, label="Conflict"),
                       Patch(facecolor="#3498db", alpha=0.6, label="Non-conflict")], fontsize=10)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(out_dir / f"gate_violin{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_classification(trained_gates, base_gates, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    results = {}
    for name, gates in [("trained", trained_gates), ("base", base_gates)]:
        if gates is None:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, class_weight="balanced")
        y_pred = cross_val_predict(clf, gates, y, cv=skf, method="predict")
        y_prob = cross_val_predict(clf, gates, y, cv=skf, method="predict_proba")[:, 1]
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        auc = roc_auc_score(y, y_prob)
        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)}
        print(f"      [{name:>7}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def process_one_model(model_name, cfg, device, max_samples):
    out_dir = OUTPUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_trained = out_dir / "trained_gates.npz"
    cache_base = out_dir / "base_gates.npz"

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {cfg['ckpt']}")
    print(f"{'='*60}")

    # tokenizer + data
    print(f"  [Data] Loading tokenizer & data ...")
    tokenizer = load_tokenizer(cfg["tokenizer"])
    samples = load_all_test_data(tokenizer, enable_thinking=cfg["enable_thinking"], max_samples=max_samples)
    labels = [s["group"] for s in samples]
    ds_labels = [s["dataset"] for s in samples]
    print(f"    Total: {len(samples)} (con={labels.count('con')}, mem={labels.count('mem')})")

    # trained model
    if cache_trained.exists():
        print(f"  [Cache] Loading trained gates from cache ...")
        d = np.load(cache_trained, allow_pickle=True)
        trained_gates = d["gates"]
    else:
        print(f"  [Model] Loading trained model ...")
        model = load_model(str(cfg["ckpt"]), cfg["code"], cfg["arch"], device)
        print(f"  [Extract] Extracting gates ...")
        trained_gates = extract_gates(model, samples, cfg["num_layers"])
        np.savez_compressed(cache_trained, gates=trained_gates,
                            labels=np.array(labels), datasets=np.array(ds_labels))
        del model; gc.collect(); torch.cuda.empty_cache()

    # base model
    if cache_base.exists():
        print(f"  [Cache] Loading base gates from cache ...")
        d = np.load(cache_base, allow_pickle=True)
        base_gates = d["gates"]
    else:
        print(f"  [Model] Loading base model ...")
        base_model = load_model(str(cfg["base"]), cfg["code"], cfg["arch"], device)
        print(f"  [Extract] Extracting base gates ...")
        base_gates = extract_gates(base_model, samples, cfg["num_layers"])
        np.savez_compressed(cache_base, gates=base_gates,
                            labels=np.array(labels), datasets=np.array(ds_labels))
        del base_model; gc.collect(); torch.cuda.empty_cache()

    # plots
    print(f"  [Plot] t-SNE ...")
    plot_tsne(trained_gates, base_gates, labels, out_dir, model_name)
    print(f"    Saved: {out_dir}/gate_tsne.pdf")

    print(f"  [Plot] Violin ...")
    plot_violin(trained_gates, labels, out_dir, model_name)
    print(f"    Saved: {out_dir}/gate_violin.pdf")

    print(f"  [Classify] Logistic regression ...")
    cls_results = run_classification(trained_gates, base_gates, labels)
    with open(out_dir / "classification_results.json", "w") as f:
        json.dump(cls_results, f, indent=2)
    print(f"    Saved: {out_dir}/classification_results.json")

    return cls_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (substring match)")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES_PER_DATASET)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = "cuda:0"

    all_results = {}
    for name, cfg in MODELS.items():
        if args.model and args.model not in name:
            continue
        res = process_one_model(name, cfg, device, args.max_samples)
        all_results[name] = res

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Trained Acc':>12} {'Trained AUC':>12} {'Base AUC':>10}")
    print(f"  {'-'*56}")
    for name, res in all_results.items():
        t = res.get("trained", {})
        b = res.get("base", {})
        print(f"  {name:<20} {t.get('accuracy','?'):>12} {t.get('auc','?'):>12} {b.get('auc','?'):>10}")

    summary_path = OUTPUT_ROOT / "all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {summary_path}")
    print(f"\n{'='*70}")
    print(f"  All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
