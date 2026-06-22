#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRQA eval — Qwen3 models, non-thinking mode.

Usage:
    python eval_qwen_nothinking.py --model qwen3_0.6B --cuda 6
    python eval_qwen_nothinking.py --model qwen3_4B   --cuda 7
"""

import os
import re
import string
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

# ─── environment ───────────────────────────────────────────────────────────────
SCRATCH = "/your/path/to/checkpoints/scratch"
os.environ["HF_HOME"] = str(SCRATCH + "hf_home")
os.environ["VLLM_USE_V1"] = "0"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ─── paths ─────────────────────────────────────────────────────────────────────
TEST_DATA_DIR = Path("/your/path/to/datasets/mrqa")

MODELS = {
    "qwen3_0.6B": "/your/path/to/models/Qwen3-0.6B",
    "qwen3_1.7B": "/your/path/to/models/Qwen3-1.7B",
    "qwen3_8B": "/your/path/to/models/Qwen3-8B"
    }

# ─── generation config ─────────────────────────────────────────────────────────
MAX_RESPONSE_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_INPUT_LENGTH = 7000

# ─── prompt templates ─────────────────────────────────────────────────────────
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


# ─── answer normalisation ──────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"\u2018", u"\u2019", u"\u00b4", u"\u0060"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(replace_underscore(s.lower())))).strip()


def extract_answer(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def acc_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ─── dataset loading ───────────────────────────────────────────────────────────
def load_and_preprocess(jsonl_path: Path, tokenizer):
    import json as _json
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_json.loads(line))

    all_input_ids = []
    all_context_answers = []
    all_qids = []

    skipped = 0
    for rec in records:
        context = rec["context"]
        question = rec["question"]
        prefix = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )},
        ]
        input_ids = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(input_ids) > MAX_INPUT_LENGTH:
            skipped += 1
            continue

        ca = rec.get("context_answer", "")
        if isinstance(ca, str):
            ca = [ca]

        all_input_ids.append(input_ids)
        all_context_answers.append(ca)
        all_qids.append(rec.get("qid", ""))

    if skipped:
        print(f"  [filter] skipped {skipped} too-long samples")
    return all_input_ids, all_context_answers, all_qids


# ─── evaluate one dataset ─────────────────────────────────────────────────────
def evaluate_dataset(dataset_path, inference_engine, tokenizer, eos_token_id, eos_token):
    print(f"\n{'─' * 60}")
    print(f"  Evaluating: {dataset_path.name}")
    print(f"{'─' * 60}")

    all_input_ids, all_context_answers, all_qids = load_and_preprocess(dataset_path, tokenizer)
    n = len(all_input_ids)
    print(f"  Samples after filtering: {n}")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        detokenize=False,
        stop_token_ids=[eos_token_id],
    )

    generations = inference_engine.generate(
        prompt_token_ids=all_input_ids,
        sampling_params=sampling_params,
    )

    total_em = total_acc = total_f1 = 0.0

    for i in range(n):
        response_token_ids = generations[i].outputs[0].token_ids
        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        if response.endswith(eos_token):
            response = response[: -len(eos_token)]

        prediction = extract_answer(response) or ""
        ground_truth = all_context_answers[i]

        total_em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truth)
        total_acc += metric_max_over_ground_truths(acc_score, prediction, ground_truth)
        total_f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truth)

    avg_em = 100.0 * total_em / n if n else 0.0
    avg_acc = 100.0 * total_acc / n if n else 0.0
    avg_f1 = 100.0 * total_f1 / n if n else 0.0

    result = {
        "dataset": dataset_path.stem,
        "n": n,
        "EM (%)": round(avg_em, 2),
        "ACC (%)": round(avg_acc, 2),
        "F1 (%)": round(avg_f1, 2),
    }

    print(f"  EM  : {avg_em:.2f}%")
    print(f"  ACC : {avg_acc:.2f}%")
    print(f"  F1  : {avg_f1:.2f}%")

    return result


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MRQA eval — Qwen3 origin, non-thinking")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--gpu_mem", type=float, default=0.5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    model_path = MODELS[args.model]

    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Model path : {model_path}")
    print(f"  CUDA       : {args.cuda}")
    print(f"  GPU mem    : {args.gpu_mem}")
    print(f"  Mode       : non-thinking")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    print(f"EOS token: {repr(eos_token)} (id={eos_token_id})")

    print("\nInitialising vLLM engine …")
    inference_engine = LLM(
        model=model_path,
        skip_tokenizer_init=False,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
        dtype=torch.bfloat16,
        max_model_len=8192,
        trust_remote_code=True,
    )
    print("vLLM engine ready.\n")

    test_files = sorted(TEST_DATA_DIR.glob("*.jsonl"))
    if not test_files:
        raise RuntimeError(f"No .jsonl files found in {TEST_DATA_DIR}")

    print(f"Found {len(test_files)} test datasets:")
    for f in test_files:
        print(f"  {f.name}")

    all_results: List[Dict[str, Any]] = []
    for test_file in test_files:
        result = evaluate_dataset(test_file, inference_engine, tokenizer, eos_token_id, eos_token)
        all_results.append(result)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  SUMMARY — {args.model} (non-thinking, MRQA)")
    print("=" * 70)
    header = f"{'Dataset':<30} {'N':>6} {'EM%':>8} {'ACC%':>8} {'F1%':>8}"
    print(header)
    print("─" * 70)
    for r in all_results:
        print(f"  {r['dataset']:<28} {r['n']:>6}"
              f" {r['EM (%)']:>8.2f} {r['ACC (%)']:>8.2f} {r['F1 (%)']:>8.2f}")

    total_n = sum(r["n"] for r in all_results)
    avg_em = sum(r["EM (%)"] * r["n"] for r in all_results) / total_n if total_n else 0.0
    avg_acc = sum(r["ACC (%)"] * r["n"] for r in all_results) / total_n if total_n else 0.0
    avg_f1 = sum(r["F1 (%)"] * r["n"] for r in all_results) / total_n if total_n else 0.0

    print("─" * 70)
    print(f"  {'MICRO-AVG':<28} {total_n:>6}"
          f" {avg_em:>8.2f} {avg_acc:>8.2f} {avg_f1:>8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

