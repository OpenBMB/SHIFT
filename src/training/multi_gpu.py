#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU GRPO Gate-Only Training

Uses DeepSpeed ZeRO-2 + torch.distributed data parallelism.
Each rank owns one vLLM inference engine and one DeepSpeed training shard.
Launch: python multi_gpu_qwen.py --nproc 4
"""

import argparse
import gc
import json
import logging
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
os.environ["VLLM_USE_V1"] = "0"

import deepspeed
import torch
import torch.distributed as dist
import swanlab
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

# ── Model code: scalar gate ──────────────────────────────────────────────────

MODEL_CODE_DIR = "/your/path/to/models/Qwen3-8B"
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
    initialize_training_process_group,
    collect_ffn_gate_activations_with_input,
    print_ffn_gate_console,
    create_ffn_gate_table,
)

# ── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ── Arg parser ───────────────────────────────────────────────────────────────

arg_parser = argparse.ArgumentParser(description="Multi-GPU GRPO Gate-Only Training")
arg_parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs to use")

# ── Hyperparameters ──────────────────────────────────────────────────────────

NUM_ITERATIONS = 200
EPISODES_PER_ITERATION = 64
GENERATIONS_PER_SAMPLE = 4
KL_COEFFICIENT = 0.001

PER_DEVICE_BATCH_SIZE = 2
GATE_LR = 1e-4
GATE_REG_COEFF = 0.01

MAX_RESPONSE_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_INPUT_LENGTH = 7000

ALL_GATE_LAYERS = list(range(36)) # this refers to the layer number of a specific LLM, for example, Qwen-3-8B has 36 layers.
TRAINABLE_GATE_LAYERS = set(ALL_GATE_LAYERS)

RUN_NAME = "gate_grpo_all_layers"
TYPE = "scalar-gate"
EXP_DIR = SCRATCH / "Qwen3-8B" / TYPE / RUN_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions based on the "
    "provided background information. "
    "The background information may be incorrect, so you should judge "
    "whether to believe yourself or to believe the background information. "
    "Provide the answer based on the given context."
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
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Policy gradient loss with pre-computed ref_log_probs from batch."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    labels_mask = batch["labels_mask"]
    advantages = batch["advantages"]
    ref_logps = batch["ref_log_probs"]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }

    labels_mask_shifted = labels_mask[..., 1:].to(torch.float)

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask_shifted

    with torch.no_grad():
        entropy = -logps.sum() / labels_mask_shifted.sum()

    policy_loss = -logps * advantages[..., 1:]
    policy_loss = policy_loss * labels_mask_shifted

    gate_reg = compute_gate_reg(policy_model)
    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len + GATE_REG_COEFF * gate_reg

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len.item(),
        "entropy": entropy.item() / total_response_len.item(),
        "gate_reg": gate_reg.item(),
    }
    return loss, metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def main(rank: int):
    args = arg_parser.parse_args()
    nproc = args.nproc

    initialize_training_process_group(rank, nproc)
    curr_cuda_device = torch.device("cuda")

    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    EPISODES_PER_ITERATION_PER_RANK = EPISODES_PER_ITERATION // dist.get_world_size()
    NUM_SAMPLES_PER_ITERATION = EPISODES_PER_ITERATION_PER_RANK // GENERATIONS_PER_SAMPLE

    logger.info("=" * 80)
    logger.info("  Multi-GPU GRPO Gate-Only — Residual-Driven (No Alpha)")
    logger.info("=" * 80)
    logger.info(f"  World size         = {dist.get_world_size()}")
    logger.info(f"  Episodes/iter/rank = {EPISODES_PER_ITERATION_PER_RANK}")
    logger.info(f"  GATE_LR            = {GATE_LR}")
    logger.info(f"  GATE_REG_COEFF     = {GATE_REG_COEFF}")
    logger.info(f"  KL_COEFFICIENT     = {KL_COEFFICIENT}")
    logger.info(f"  ALL_GATE_LAYERS    = all {len(ALL_GATE_LAYERS)} layers")
    logger.info(f"  REWARD             = format + faithful (no alpha aux)")

    # ── DeepSpeed config (adjusted for multi-GPU) ─────────────────────────

    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION_PER_RANK // PER_DEVICE_BATCH_SIZE,
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
        "gradient_accumulation_steps": EPISODES_PER_ITERATION_PER_RANK // PER_DEVICE_BATCH_SIZE,
    }

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # ── Load datasets ────────────────────────────────────────────────────

    if dist.get_rank() != 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])

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

    if dist.get_rank() == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.barrier(device_ids=[torch.cuda.current_device()])

    orig_train_size = len(train_dataset)
    train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

    logger.info(f"  Train: {orig_train_size} total, shard={len(train_dataset)}, Valid: {len(valid_dataset)}")

    # ── Initialize models ────────────────────────────────────────────────

    ffn_config = Qwen3Config.from_pretrained(MODEL_CODE_DIR)
    model_kwargs = {
        "config": ffn_config,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": torch.cuda.current_device(),
    }

    policy_model = Qwen3ForCausalLM.from_pretrained(MODEL_WEIGHTS_DIR, **model_kwargs)
    reference_model = Qwen3ForCausalLM.from_pretrained(MODEL_WEIGHTS_DIR, **model_kwargs)

    # ── Freeze LLM, only train gate layers ───────────────────────────────

    for name, param in policy_model.named_parameters():
        if "ffn_gate" in name:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            param.requires_grad = layer_idx in TRAINABLE_GATE_LAYERS
        else:
            param.requires_grad = False

    num_total = sum(p.numel() for p in policy_model.parameters())
    num_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    num_frozen = num_total - num_trainable

    logger.info(f"\n  Parameter Summary:")
    logger.info(f"    Total:     {num_total:>12,}")
    logger.info(f"    Frozen:    {num_frozen:>12,} (LLM)")
    logger.info(f"    Trainable: {num_trainable:>12,} (ffn_gate only)")

    if dist.get_rank() == 0:
        gate_params_detail = [
            (name, param.numel())
            for name, param in policy_model.named_parameters()
            if param.requires_grad
        ]
        logger.info(f"\n  Trainable gate parameters ({len(gate_params_detail)} tensors):")
        for name, n in gate_params_detail:
            logger.info(f"    {name}: {n}")

    # ── Gradient checkpointing ───────────────────────────────────────────

    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── DeepSpeed init ───────────────────────────────────────────────────

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

    dist.barrier(device_ids=[torch.cuda.current_device()])

    # ── vLLM engine (one per rank) ───────────────────────────────────────

    if dist.get_rank() != 0:
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

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
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────

    sampler_rng = np.random.default_rng(seed=42)
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        logger.info(f"  Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        policy_model.load_checkpoint(str(ckpt_path / "deepspeed"))
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)
        logger.info(f"  Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter, disable=dist.get_rank() != 0):
            _ = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
    else:
        logger.info("  Starting from scratch (iteration 0)")

    # ── SwanLab (rank 0 only) ────────────────────────────────────────────

    if dist.get_rank() == 0:
        swanlab.login(api_key="xxx") # change to your swanlab api key
        swanlab_run = swanlab.init(
            project="ffn",
            workspace="xxx", # change to your swanlab username
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
                "world_size": dist.get_world_size(),
            },
        )

    # ── Training loop ────────────────────────────────────────────────────

    for iteration in trange(begin_iter, NUM_ITERATIONS, initial=begin_iter,
                            total=NUM_ITERATIONS, disable=dist.get_rank() != 0):
        logger.info(f"\nIteration {iteration}/{NUM_ITERATIONS}")
        metrics = {}

        # ── Evaluation (rank 0 only) ─────────────────────────────────────
        eval_stats = None
        if iteration % 25 == 0 and dist.get_rank() == 0:
            logger.info("  Evaluating on valid set ...")
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
        dist.barrier(device_ids=[torch.cuda.current_device()])

        # ── Sample batch ─────────────────────────────────────────────────
        indices = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # ── Generate responses (vLLM) ────────────────────────────────────
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

        logger.info(f"Generated {len(all_generations)} responses in {time.time() - gen_time:.1f}s")

        inference_engine.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # ── Create training episodes ─────────────────────────────────────
        episodes, episodes_stats = create_training_episodes(
            samples, all_generations, all_finish_reasons,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes, episodes_stats=episodes_stats,
            exp_dir=EXP_DIR, tokenizer=tokenizer, iteration=iteration,
        )

        # ── Prepare model inputs ─────────────────────────────────────────
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=curr_cuda_device,
        )

        # ── Pre-compute reference log probs ──────────────────────────────
        logger.info("Computing reference logprobs...")
        reference_model.module.to(curr_cuda_device)
        reference_model.eval()

        with torch.no_grad():
            ref_log_probs = []
            for i in trange(
                0, EPISODES_PER_ITERATION_PER_RANK, PER_DEVICE_BATCH_SIZE,
                desc="Ref logprobs", disable=dist.get_rank() != 0,
            ):
                batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}
                ref_log_probs.append(compute_token_log_probs(reference_model, batch, TEMPERATURE))
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        logger.info("Moving reference model back to CPU")
        reference_model.module.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # ── Training ─────────────────────────────────────────────────────
        policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        for i in trange(
            0, EPISODES_PER_ITERATION_PER_RANK, PER_DEVICE_BATCH_SIZE,
            desc="GA", disable=dist.get_rank() != 0,
        ):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
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
            policy_model.step()

        logger.info(f"Training step took {time.time() - train_time:.1f}s")

        # ── Update vLLM weights ──────────────────────────────────────────
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # ── Logging (rank 0 only) ────────────────────────────────────────
        if dist.get_rank() == 0:
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

            # ── Gate activation stats ────────────────────────────────────
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

            # ── Flag group metrics ───────────────────────────────────────
            flag_true_r = train_metrics.get("flag/flag_true/rewards", float("nan"))
            flag_false_r = train_metrics.get("flag/flag_false/rewards", float("nan"))
            flag_true_f = train_metrics.get("flag/flag_true/faithful_reward", float("nan"))
            flag_false_f = train_metrics.get("flag/flag_false/faithful_reward", float("nan"))
            flag_gap = (
                flag_true_r - flag_false_r
                if flag_true_r == flag_true_r and flag_false_r == flag_false_r
                else float("nan")
            )

            logger.info(f"\n{'─' * 60}")
            logger.info(f"  FLAG Group Reward (iter={iteration})")
            logger.info(f"{'─' * 60}")
            logger.info(f"  flag=True  (ConAcc proxy): reward={flag_true_r:.4f}, faithful={flag_true_f:.4f}")
            logger.info(f"  flag=False (MemAcc proxy): reward={flag_false_r:.4f}, faithful={flag_false_f:.4f}")
            logger.info(f"  Gap (T-F): {flag_gap:+.4f}")
            logger.info(f"{'─' * 60}")

            logs.update({
                "flag/true_reward": flag_true_r,
                "flag/false_reward": flag_false_r,
                "flag/reward_gap": flag_gap,
            })

            swanlab.log(logs)

            logger.info(f"  KEY METRICS: loss={train_metrics.get('loss', 'N/A'):.4f}, "
                        f"reward={train_metrics.get('rewards', 'N/A'):.4f}, "
                        f"gate_reg={train_metrics.get('gate_reg', 'N/A'):.6f}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if iteration % 100 == 0 and iteration != 0:
            ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"

            logger.info("Saving HF model")
            if dist.get_rank() == 0:
                policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"))
                src_dir = Path(MODEL_CODE_DIR)
                for fname in ["modeling_qwen3.py", "configuration_qwen3.py"]:
                    src = src_dir / fname
                    if src.exists():
                        shutil.copy2(src, ckpt_dir / "hf_model" / fname)
            dist.barrier(device_ids=[torch.cuda.current_device()])

            logger.info("Saving DeepSpeed checkpoint")
            policy_model.save_checkpoint(str(ckpt_dir / "deepspeed"))
            dist.barrier(device_ids=[torch.cuda.current_device()])

            logger.info(f"  Checkpoint saved: {ckpt_dir}")

    if dist.get_rank() == 0:
        swanlab.finish()

    dist.destroy_process_group()
    logger.info("Training complete!")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if args.nproc > n_gpus:
        raise ValueError(f"Requested {args.nproc} processes, but only {n_gpus} GPUs are available.")

    if args.nproc == 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.nproc)
