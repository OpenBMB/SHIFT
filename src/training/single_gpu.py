"""
We built this file based on https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1.ipynb
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import json
import os
import re
import shutil
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ── Environment ──────────────────────────────────────────────────────────────

SCRATCH = Path("/your/path/to/save/checkpoints/scratch")
os.environ["HF_HOME"] = str(SCRATCH / "hf_home")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import socket


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


os.environ["MASTER_PORT"] = str(find_free_port())

import deepspeed
import torch
import swanlab
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel

os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams

# ── Model code: scalar gate ──────────────────────────────────────────────────

MODEL_CODE_DIR = "/your/path/to/models/Qwen3-1.7B" # this refers to the modied model codes for SHIFT
MODEL_WEIGHTS_DIR = MODEL_CODE_DIR

sys.path.insert(0, MODEL_CODE_DIR)
from configuration_qwen3 import Qwen3Config
from modeling_qwen3 import Qwen3ForCausalLM

# ── Utils from training framework ───────────────────────────────────────────

sys.path.insert(0, "/your/path/to/save/checkpoints")
from utils import (
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_last_checkpoint,
    prepare_model_inputs,
    load_model_into_vllm,
    collect_ffn_gate_activations_with_input,
    print_ffn_gate_console,
    create_ffn_gate_table,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_ITERATIONS = 200
EPISODES_PER_ITERATION = 64
GENERATIONS_PER_SAMPLE = 4
KL_COEFFICIENT = 0.001

PER_DEVICE_BATCH_SIZE = 4
GATE_LR = 1e-4
GATE_REG_COEFF = 0.01

MAX_RESPONSE_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_INPUT_LENGTH = 7000

ALL_GATE_LAYERS = list(range(28)) # this refers to the layer number of a specific LLM, for example, Qwen-3-1.7B has 28 layers.
TRAINABLE_GATE_LAYERS = set(ALL_GATE_LAYERS)

RUN_NAME = "gate_grpo_all_layers_nothinking"
TYPE = "scalar-gate"
EXP_DIR = SCRATCH / "Qwen3-1.7B" / TYPE / RUN_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

# ── DeepSpeed config ─────────────────────────────────────────────────────────

deepspeed_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 0},
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": GATE_LR,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": True,
        },
    },
}

ref_deepspeed_config = {
    "bf16": {"enabled": True},
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
}

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions based on the "
    "provided background information."
    "The background information may be incorrect, so you should judge "
    "whether to believe yourself or to believe the background information."
    "You should think about the reasoning process and then provide the "
    "answer based on the given context."
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

# ── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE_DIR, trust_remote_code=False)
EOS_TOKEN_ID = tokenizer.eos_token_id
EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)


# ══════════════════════════════════════════════════════════════════════════════
# Reward Functions
# ══════════════════════════════════════════════════════════════════════════════


def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def extract_answer(text: str):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def format_reward_func(completion: str) -> float:
    if completion.endswith(EOS_TOKEN):
        completion = completion[: -len(EOS_TOKEN)]
    regex = r"<answer>([\s\S]*?)<\/answer>"
    match = re.search(regex, completion, re.DOTALL)
    if match is None:
        return 0.0
    answer_content = match.group(1).strip()
    return 1.0 if len(answer_content) > 0 else 0.5


def faithful_reward_func(
    completion: str,
    answer: Union[str, List[str]],
    flag: bool = True,
    parametric_answer: Union[str, List[str]] = None,
) -> float:
    target = answer if flag else parametric_answer
    if target is None:
        return 0.0
    try:
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]
        extracted = extract_answer(completion)
        if extracted is None:
            return 0.0
        norm_pred = normalize_answer(extracted)
        targets = target if isinstance(target, list) else [target]
        for t in targets:
            if norm_pred == normalize_answer(t):
                return 1.0
        return 0.0
    except Exception:
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Pure correctness reward — no alpha, no auxiliary signal."""
    format_r = format_reward_func(completion)
    flag = sample.get("flag", True)
    faithful_r = faithful_reward_func(
        completion=completion,
        answer=sample["context_answer"],
        flag=flag,
        parametric_answer=sample.get("parametric_answer"),
    )
    total_reward = format_r + faithful_r
    return total_reward, {
        "format_reward": format_r,
        "faithful_reward": faithful_r,
    }


# ── Data preprocessing ───────────────────────────────────────────────────────


def preprocess_example(example: Dict[str, Any]):
    question = example["question"]
    cf_context = example["context"] if example["cf_context"] == "" else example["cf_context"]
    flag = not example["is_parametric_answer_right"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT_TEMPLATE.format(context=cf_context, question=question)},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        prefix, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "qid": example["qid"],
        "question": question,
        "cf_context": cf_context,
        "context_answer": example["context_answer"],
        "parametric_answer": example["parametric_answer"],
        "flag": flag,
    }


# ── Training episodes ────────────────────────────────────────────────────────


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []
    stats = {
        "response_lengths": [], "rewards": [], "non_stop_rate": [],
    }

    for sample_idx, (sample, group_indices) in enumerate(zip(samples, groups)):
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        response_token_ids = [all_generations[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

        rewards_and_metrics = [compute_reward(resp, sample) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        response_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        advantages = [
            [resp_adv] * len(resp)
            for resp_adv, resp in zip(response_advantages, response_token_ids)
        ]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])

        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

        sample_flag = sample.get("flag", True)
        flag_key = "flag_true" if sample_flag else "flag_false"
        stats.setdefault(f"flag/{flag_key}/rewards", []).append(float(np.mean(rewards)))
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"flag/{flag_key}/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }
    return episodes, stats


# ── Policy gradient loss ─────────────────────────────────────────────────────


def compute_gate_reg(policy_model: Union[DeepSpeedEngine, PreTrainedModel]) -> torch.Tensor:
    model = policy_model.module if hasattr(policy_model, "module") else policy_model
    reg = torch.tensor(0.0, device="cuda")
    n = 0
    for layer in model.model.layers:
        if hasattr(layer, "ffn_gate") and layer.ffn_gate.bias.requires_grad:
            reg = reg + layer.ffn_gate.bias.pow(2).sum()
            reg = reg + layer.ffn_gate.weight.pow(2).sum()
            n += 1
    return reg / max(n, 1)


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    advantages = batch["advantages"]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": batch["labels_mask"],
    }

    labels_mask = (labels[..., 1:] != -100).float()

    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, model_inputs, TEMPERATURE)

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask
    entropy = -logps.sum() / labels_mask.sum()
    policy_loss = -logps * advantages[..., 1:]
    policy_loss = policy_loss * labels_mask

    gate_reg = compute_gate_reg(policy_model)
    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len + GATE_REG_COEFF * gate_reg

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
        "gate_reg": gate_reg.item(),
    }
    return loss, metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("  Exp09: GRPO Gate-Only — Residual-Driven (No Alpha)")
    print("=" * 80)
    print(f"  GATE_LR           = {GATE_LR}")
    print(f"  GATE_REG_COEFF    = {GATE_REG_COEFF}")
    print(f"  KL_COEFFICIENT    = {KL_COEFFICIENT}")
    print(f"  ALL_GATE_LAYERS   = all {len(ALL_GATE_LAYERS)} layers")
    print(f"  REWARD            = format + faithful (no alpha aux)")

    # ── Load datasets ──────────────────────────────────────────────────────
    train_dataset = load_dataset(
        "json", data_files="/your/path/to/project/datasets/train/train.json", split="train"
    )
    valid_dataset = load_dataset(
        "json", data_files="/your/path/to/project/datasets/valid/valid.json", split="train"
    )

    train_dataset = train_dataset.map(preprocess_example, num_proc=6)
    valid_dataset = valid_dataset.map(preprocess_example, num_proc=6)

    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= MAX_INPUT_LENGTH, num_proc=6)
    valid_dataset = valid_dataset.filter(lambda x: len(x["input_ids"]) <= MAX_INPUT_LENGTH, num_proc=6)

    print(f"  Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    # ── Initialize models ──────────────────────────────────────────────────
    ffn_config = Qwen3Config.from_pretrained(MODEL_CODE_DIR)
    model_kwargs = {
        "config": ffn_config,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": 0,
    }

    policy_model = Qwen3ForCausalLM.from_pretrained(MODEL_WEIGHTS_DIR, **model_kwargs)
    reference_model = Qwen3ForCausalLM.from_pretrained(MODEL_WEIGHTS_DIR, **model_kwargs)

    # ── Freeze LLM, only train battleground layers' gate ──────────────────
    for name, param in policy_model.named_parameters():
        if "ffn_gate" in name:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            param.requires_grad = layer_idx in TRAINABLE_GATE_LAYERS
        else:
            param.requires_grad = False

    num_total = sum(p.numel() for p in policy_model.parameters())
    num_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    num_frozen = num_total - num_trainable

    print(f"\n  Parameter Summary:")
    print(f"    Total:     {num_total:>12,}")
    print(f"    Frozen:    {num_frozen:>12,} (LLM)")
    print(f"    Trainable: {num_trainable:>12,} (ffn_gate only)")

    gate_params_detail = [
        (name, param.numel())
        for name, param in policy_model.named_parameters()
        if param.requires_grad
    ]
    print(f"\n  Trainable gate parameters ({len(gate_params_detail)} tensors):")
    for name, n in gate_params_detail:
        print(f"    {name}: {n}")

    # ── Gradient checkpointing ────────────────────────────────────────────
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── DeepSpeed init ─────────────────────────────────────────────────────
    gate_params = [p for n, p in policy_model.named_parameters() if p.requires_grad]
    param_groups = [{"params": gate_params, "lr": GATE_LR}]

    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=param_groups,
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )
    reference_model.module.cpu()

    # ── vLLM engine ────────────────────────────────────────────────────────
    inference_engine = LLM(
        model=MODEL_CODE_DIR,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        swap_space=1,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=8192,
        enable_sleep_mode=True,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        print(f"  Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        policy_model.load_checkpoint(str(ckpt_path / "deepspeed"))
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)
        print(f"  Will continue from iteration {begin_iter}")
    else:
        print(f"  Starting from scratch (iteration 0)")

    # ── SwanLab ────────────────────────────────────────────────────────────
    swanlab.login(api_key="xxxx") # change to your swanlab api key
    swanlab_run = swanlab.init(
        project="ffn",
        workspace="xxx", # change to your swanlab user name
        experiment_name=RUN_NAME,
        config={
            "model_code": MODEL_CODE_DIR,
            "gate_type": TYPE,
            "gate_lr": GATE_LR,
            "gate_reg_coeff": GATE_REG_COEFF,
            "frozen_llm": True,
            "reward": "format + faithful",
            "num_iterations": NUM_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
            "kl_coefficient": KL_COEFFICIENT,
            "temperature": TEMPERATURE,
            "trainable_params": num_trainable,
            "trainable_layers": ALL_GATE_LAYERS,
            "resume_from": begin_iter,
        },
    )

    # ── Training loop ──────────────────────────────────────────────────────
    for iteration in trange(begin_iter, NUM_ITERATIONS, initial=begin_iter, total=NUM_ITERATIONS):
        print(f"\nIteration {iteration}/{NUM_ITERATIONS}")
        metrics = {}

        # ── Evaluation ─────────────────────────────────────────────────────
        eval_stats = None
        if iteration % 25 == 0:
            print("  Evaluating on valid set ...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=valid_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.6, max_tokens=1024, n=1,
                    detokenize=False, stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(completion, sample),
            )
            eval_table = dump_episodes(
                episodes=eval_episodes, episodes_stats=eval_stats,
                exp_dir=EXP_DIR, tokenizer=tokenizer,
                iteration=iteration, is_eval=True,
            )
            swanlab.log({"eval/episodes": eval_table, "iteration": iteration})

        # ── Sample batch ───────────────────────────────────────────────────
        num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        # ── Generate responses (vLLM) ──────────────────────────────────────
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # ── Create training episodes ──────────────────────────────────────
        episodes, episodes_stats = create_training_episodes(
            samples, all_generations, all_finish_reasons,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes, episodes_stats=episodes_stats,
            exp_dir=EXP_DIR, tokenizer=tokenizer, iteration=iteration,
        )

        # ── Training ───────────────────────────────────────────────────────
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device="cuda",
        )

        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        for i in trange(0, EPISODES_PER_ITERATION, PER_DEVICE_BATCH_SIZE, desc="GA"):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
            )

            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            policy_model.backward(loss, scale_wrt_gas=False)
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()
            policy_model.step()

        # ── Update vLLM weights ────────────────────────────────────────────
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # ── Logging ────────────────────────────────────────────────────────
        train_metrics = {k: np.mean(v) for k, v in metrics.items() if v and None not in v}
        train_metrics["learning_rate"] = policy_model.get_lr()[0]

        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            eval_metrics = {k: np.mean(v) for k, v in eval_stats.items() if v and None not in v}
            logs.update({f"eval/{k}": v for k, v in eval_metrics.items()})

        # ── Gate activation stats ──────────────────────────────────────────
        policy_model.eval()
        ffn_gate_act = collect_ffn_gate_activations_with_input(
            model=policy_model.module,
            input_ids=model_inputs["input_ids"][:1],
            attention_mask=model_inputs["attention_mask"][:1],
        )
        policy_model.train()

        if ffn_gate_act["layer_indices"]:
            print_ffn_gate_console(ffn_gate_act, iteration)
            logs.update({
                "ffn_gate/mean": round(float(np.mean(ffn_gate_act["gate_mean"])), 6),
                "ffn_gate/min": round(float(np.min(ffn_gate_act["gate_min"])), 6),
                "ffn_gate/max": round(float(np.max(ffn_gate_act["gate_max"])), 6),
                **{
                    f"ffn_gate/layer_{li:02d}": round(float(ffn_gate_act["gate_mean"][i]), 6)
                    for i, li in enumerate(ffn_gate_act["layer_indices"])
                },
            })
            ffn_gate_table = create_ffn_gate_table(ffn_gate_act, iteration)
            logs[f"ffn_gate/stats_iter_{iteration:06d}"] = ffn_gate_table

        # ── Flag group metrics ─────────────────────────────────────────────
        flag_true_r = train_metrics.get("flag/flag_true/rewards", float("nan"))
        flag_false_r = train_metrics.get("flag/flag_false/rewards", float("nan"))
        flag_true_f = train_metrics.get("flag/flag_true/faithful_reward", float("nan"))
        flag_false_f = train_metrics.get("flag/flag_false/faithful_reward", float("nan"))
        flag_gap = (
            flag_true_r - flag_false_r
            if flag_true_r == flag_true_r and flag_false_r == flag_false_r
            else float("nan")
        )

        print(f"\n{'─' * 60}")
        print(f"  FLAG Group Reward (iter={iteration})")
        print(f"{'─' * 60}")
        print(f"  flag=True  (ConAcc proxy): reward={flag_true_r:.4f}, faithful={flag_true_f:.4f}")
        print(f"  flag=False (MemAcc proxy): reward={flag_false_r:.4f}, faithful={flag_false_f:.4f}")
        print(f"  Gap (T-F): {flag_gap:+.4f}")
        print(f"{'─' * 60}")

        logs.update({
            "flag/true_reward": flag_true_r,
            "flag/false_reward": flag_false_r,
            "flag/reward_gap": flag_gap,
        })

        swanlab.log(logs)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if iteration % 50 == 0 and iteration != 0:
            ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"
            policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"))
            src_dir = Path(MODEL_CODE_DIR)
            for fname in ["modeling_qwen3.py", "configuration_qwen3.py"]:
                src = src_dir / fname
                if src.exists():
                    shutil.copy2(src, ckpt_dir / "hf_model" / fname)
            policy_model.save_checkpoint(str(ckpt_dir / "deepspeed"))
            print(f"  Checkpoint saved: {ckpt_dir}")

        print(f"  KEY METRICS: loss={train_metrics.get('loss', 'N/A'):.4f}, "
              f"reward={train_metrics.get('rewards', 'N/A'):.4f}, "
              f"gate_reg={train_metrics.get('gate_reg', 'N/A'):.6f}")

    swanlab.finish()
    print("\n  Training complete!")


if __name__ == "__main__":
    main()
