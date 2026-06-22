"""
We built this file based on https://github.com/McGill-NLP/nano-aha-moment/blob/main/utils.py
"""


from datetime import timedelta
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import swanlab
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
DEFAULT_PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."


def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    prefix = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        advantages: List of lists of advantage values, matching response_token_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": [], "labels_mask": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len - seq_len)
        advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len
        assert len(labels_mask) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)
        inputs["labels_mask"].append(labels_mask)

    # Convert to tensors
    return {
        k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
        for k, v in inputs.items()
    }


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425

    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing the logits
        index: Tensor of shape (batch_size, seq_len) containing the indices to gather

    Returns:
        Tensor of shape (batch_size, seq_len) containing the log probabilities for the
        specified indices

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_token_log_probs(
    model: Union[DeepSpeedEngine, PreTrainedModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (either DeepSpeed-wrapped or regular HuggingFace model)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1]
    shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the model on a test dataset by generating responses and computing rewards.

    Args:
        inference_engine: The sglang Engine instance used for text generation
        test_dataset: Dataset containing test samples
        tokenizer: Tokenizer for decoding generated token IDs
        eos_token: End of sequence token string
        eval_sampling_params: Dictionary of parameters for controlling the generation process
        reward_func: Function that computes rewards for generated responses. Takes a response
            string and sample dict as input, returns a tuple of (overall_reward, reward_components)

    Returns:
        Dictionary containing evaluation statistics:
            - response_lengths: List of token counts for each generated response
            - rewards: List of overall reward values for each response
            - non_stop_rate: List of booleans indicating if generation ended for non-stop reason
            - reward_metrics/*: Lists of individual reward component values, prefixed with
              "reward_metrics/"
        episodes: Dictionary containing:
            - all_query_token_ids: List of query token IDs for each episode
            - all_response_token_ids: List of response token IDs for each episode

    Example:
        >>> episodes, episodes_stats = evaluate_on_test_set(
        ...     inference_engine=engine,
        ...     test_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     eos_token="</s>",
        ...     eval_sampling_params={"temperature": 0.7, "max_tokens": 100},
        ...     reward_func=compute_rewards
        ... )
        >>> print(f"Average reward: {episodes_stats['rewards']:.3f}")
    """
    generations = inference_engine.generate(
        [{"prompt_token_ids": ids} for ids in test_dataset["input_ids"]],
        sampling_params=eval_sampling_params,
    )

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    for i, sample in enumerate(test_dataset):
        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        reward, reward_components = reward_func(response, sample)

        all_query_token_ids.append(query_token_ids)
        all_responses_token_ids.append(response_token_ids)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))
        for k, v in reward_components.items():
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }

    return episodes, metrics


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
    do_save: bool = True,
) -> swanlab.echarts.Table:
    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if not is_eval and rank == 0:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")

    if is_eval:
        episodes_dir = exp_dir / "eval_episodes"
    else:
        episodes_dir = exp_dir / "episodes"
    if dist.is_initialized():
        episodes_dir = episodes_dir / f"rank_{rank:02d}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    
    headers = ["query", "response", "reward", "response_length"]
   
    rows = []
    for i in range(len(query_texts)):
        rows.append([
            query_texts[i],
            response_texts[i],
            float(rewards[i]),  
            int(response_lengths[i])
        ])
    
   
    table = swanlab.echarts.Table()
    table.add(headers, rows)

    if not do_save:
        return table

    with open(episodes_dir / f"eps_{iteration:06d}.json", "w") as f:
        json.dump(
            [
                {
                    "query": query_texts[i],
                    "response": response_texts[i],
                    "reward": rewards[i],
                }
                for i in range(len(query_texts))
            ],
            f,
        )

    return table


def find_last_checkpoint(exp_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("ckpt_*"))
    # Filter out directories that don't have a deepspeed subdirectory
    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    ckpt_iter = int(ckpt_path.stem.split("_")[-1])
    return ckpt_path, ckpt_iter


def load_model_into_vllm(model: Union[DeepSpeedEngine, PreTrainedModel], llm: LLM) -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (Union[DeepSpeedEngine, PreTrainedModel]): The source model to copy weights from.
            Can be either a DeepSpeed-wrapped model or a regular HuggingFace PreTrainedModel.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


def initialize_training_process_group(rank: int, world_size: int):
    """
    Initialize the PyTorch distributed process group for multi-GPU training using NCCL backend.

    This function sets up the distributed training environment by:
    1. Setting the CUDA device for the current process
    2. Initializing the process group with NCCL backend
    3. Creating a barrier to ensure all processes are synchronized

    Args:
        rank (int): The rank of the current process (0 to world_size-1)
        world_size (int): Total number of processes participating in the distributed training

    Note:
        - The function uses a free port on localhost for process group initialization
        - A timeout of 1800 seconds (30 minutes) is set for process group initialization
    """
    master_addr = "localhost"
    master_training_port = int(os.environ.get("MASTER_PORT", "8237"))

    # os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    # os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)

    if rank == 0:
        print(
            f"{'#' * 80}\n" f"# Initializing the training NCCL PG with\n" f"# world_size={world_size} \n" f"{'#' * 80}"
        )

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_training_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=1),
    )
    dist.barrier(device_ids=[rank])
    print(
        f"Rank{rank}: training NCCL PG initialized. "
        f"(world_size={world_size}, local_rank={rank}, gpu_id={torch.cuda.current_device()})"
    )


def clean_up_checkpoints(
    exp_dir: Path, keep_every_n_steps: Optional[int] = None, exclude: Optional[List[Path]] = None
) -> None:
    """
    Clean up checkpoint directories by removing unnecessary files and directories.

    This function manages checkpoint storage by:
    1. Keeping only essential model files (hf_model) in checkpoints that are multiples of keep_every_n_steps
    2. Removing all other checkpoints that are not in the exclude list
    3. Preserving checkpoints that are in the exclude list regardless of their iteration number

    Args:
        exp_dir (Path): The experiment directory containing the checkpoints
        keep_every_n_steps (Optional[int]): If specified, keeps checkpoints that are multiples of this number.
            For these checkpoints, only the hf_model directory is preserved.
        exclude (Optional[List[Path]]): List of checkpoint paths to exclude from cleanup.
            These checkpoints will be preserved regardless of their iteration number.

    Example:
        >>> clean_up_checkpoints(
        ...     exp_dir=Path("experiments/run1"),
        ...     keep_every_n_steps=1000,
        ...     exclude=[Path("experiments/run1/checkpoints/ckpt_5000")]
        ... )
        # This will:
        # - Keep checkpoints 1000, 2000, 3000, etc. (only hf_model directory)
        # - Keep checkpoint 5000 completely (all files)
        # - Remove all other checkpoints
    """
    if exclude is None:
        exclude = []

    checkpoint_dir = exp_dir / "checkpoints"
    for ckpt in checkpoint_dir.glob("ckpt_*"):
        if keep_every_n_steps is None or ckpt in exclude:
            continue

        ckpt_iter = int(ckpt.stem.split("_")[-1])
        if ckpt_iter % keep_every_n_steps == 0:
            # Remove non-hf_model files and dirs
            removed_files_and_dirs = []
            for file in ckpt.iterdir():
                if file.name not in ["hf_model"]:
                    try:
                        removed_files_and_dirs.append(file.name)
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                    except Exception as e:
                        print(f"Error removing {file}: {e}")
            if len(removed_files_and_dirs) > 0:
                print(f"Removed non-hf_model files and dirs: of checkpoint {ckpt.name}")

            continue

        print(f"Removing checkpoint {ckpt}")
        shutil.rmtree(ckpt)


def fix_oov_logits_processor(inference_engine: LLM):
    # https://github.com/issues/recent?issue=vllm-project%7Cvllm%7C13175
    # Qwen and some other models come with a few hundred extra out-of-vocab tokens that can be used for
    # fine-tuning in case new special domain-specific tokens are required.

    # Sampling the OOV token will trigger an error:
    # ValueError: Token id 151791 is out of vocabulary
    # So we mask them using process_token
    # fix_oov # remove asap when this is fixed in vllm, it is dirty and even logit processors are not supported in engine v1 of vllm

    tokenizer_vocab_size = len(inference_engine.get_tokenizer().get_vocab())

    def fix_oov(token_ids, logits):
        logits[tokenizer_vocab_size:] = -float("inf")
        return logits

    return fix_oov


def close_to_zero(tensor: torch.Tensor, mask: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    """
    Computes the number of values in the tensor that are close to zero.
    """
    close_to_zero_mask = torch.abs(tensor) < threshold
    num_close_to_zero = (close_to_zero_mask * mask).sum()
    return num_close_to_zero


def collect_ffn_gate_activations_with_input(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, Any]:
   
    activations: Dict[str, Any] = {
        'layer_indices': [],
        'gate_mean': [],
        'gate_std':  [],
        'gate_min':  [],
        'gate_max':  [],
        'gate_dim_std': [],
        'gate_active_ratio': [],
        'is_vector_gate': [],
    }

    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return activations

    layer_raw_outputs: Dict[int, torch.Tensor] = {}

    def _infer_gate_temp(gate_module: torch.nn.Module) -> float:
        
        cfg = getattr(model, 'config', None)
        cfg_temp = getattr(cfg, 'ffn_gate_temperature', None) if cfg is not None else None
        if cfg_temp is not None:
            try:
                return float(cfg_temp)
            except (TypeError, ValueError):
                pass

        
        out_dim = gate_module.weight.shape[0] if hasattr(gate_module, 'weight') else None
        return 1.0 if out_dim == 1 else 8.0

    def _make_hook(layer_idx: int):
        def hook_fn(module, args, output):
            gate_temp = _infer_gate_temp(module)
            gate_val = 2.0 * torch.sigmoid(output.detach().cpu() / gate_temp)
            layer_raw_outputs[layer_idx] = gate_val
        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if not getattr(layer, 'ffn_output_gate', False):
            continue
        if not hasattr(layer, 'ffn_gate'):
            continue
        hook = layer.ffn_gate.register_forward_hook(_make_hook(layer_idx))
        hooks.append(hook)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    for layer_idx in sorted(layer_raw_outputs.keys()):
        gate = layer_raw_outputs[layer_idx]  # [bsz, seq_len, gate_dim]
        gate_dim = gate.shape[-1]
        is_vector = gate_dim > 1

        activations['layer_indices'].append(layer_idx)
        activations['gate_mean'].append(gate.mean().item())
        activations['gate_std'].append(gate.std().item())
        activations['gate_min'].append(gate.min().item())
        activations['gate_max'].append(gate.max().item())
        activations['is_vector_gate'].append(is_vector)

        if is_vector:
            active_ratio = (gate > 1.0).float().mean().item()
        else:
            dim_std = 0.0
            active_ratio = (gate > 1.0).float().mean().item()

        activations['gate_dim_std'].append(dim_std)
        activations['gate_active_ratio'].append(active_ratio)

    return activations


def collect_gate_statistics(model: PreTrainedModel) -> Dict[str, Any]:
   
    gate_stats = {
        'bias_mean': [],
        'bias_std': [],
        'bias_min': [],
        'bias_max': [],
        'weight_mean': [],
        'weight_std': [],
        'weight_norm': [],
        'gate_score_mean': [],  # these are just estimated numbers to observe during training, please run the analysis scripts for true gates 
        'layer_indices': [],
    }
    
    config = model.config
    if not hasattr(config, 'retrieval_heads_config'):
        return gate_stats
    
    retrieval_heads_config = config.retrieval_heads_config
    if retrieval_heads_config is None:
        return gate_stats
    
    retrieval_heads_by_layer = retrieval_heads_config.get('retrieval_heads', {})
    
    for layer_idx_str, head_indices in retrieval_heads_by_layer.items():
        layer_idx = int(layer_idx_str)
        attn = model.model.layers[layer_idx].self_attn
        
        if not hasattr(attn, 'retrieval_gate_proj') or attn.retrieval_gate_proj is None:
            continue
        
        gate_proj = attn.retrieval_gate_proj
        
        
        if gate_proj.bias is not None:
            bias = gate_proj.bias.detach().cpu()
            gate_stats['bias_mean'].append(bias.mean().item())
            gate_stats['bias_std'].append(bias.std().item())
            gate_stats['bias_min'].append(bias.min().item())
            gate_stats['bias_max'].append(bias.max().item())
            
            # these are just estimated numbers to observe during training, please run the analysis scripts for true gates 
            gate_score = 1.0 + torch.tanh(bias)
            gate_stats['gate_score_mean'].append(gate_score.mean().item())
        else:
            gate_stats['bias_mean'].append(0.0)
            gate_stats['bias_std'].append(0.0)
            gate_stats['bias_min'].append(0.0)
            gate_stats['bias_max'].append(0.0)
            gate_stats['gate_score_mean'].append(1.0)
        
        
        if gate_proj.weight is not None:
            weight = gate_proj.weight.detach().cpu()
            gate_stats['weight_mean'].append(weight.mean().item())
            gate_stats['weight_std'].append(weight.std().item())
            gate_stats['weight_norm'].append(weight.norm().item())
        else:
            gate_stats['weight_mean'].append(0.0)
            gate_stats['weight_std'].append(0.0)
            gate_stats['weight_norm'].append(0.0)
        
        gate_stats['layer_indices'].append(layer_idx)
    
    return gate_stats


def create_gate_table(gate_stats: Dict[str, Any], iteration: int) -> swanlab.echarts.Table:
   
    if not gate_stats['layer_indices']:
        
        table = swanlab.echarts.Table()
        table.add(
            ['iteration', 'layer', 'info'],
            [[iteration, 0, 'No retrieval gates found']]
        )
        return table
    
    
    headers = [
        'iteration',
        'layer',
        'bias_mean',
        'bias_std',
        'bias_min',
        'bias_max',
        'weight_mean',
        'weight_std',
        'weight_norm',
        'gate_score_mean'
    ]
    
    
    rows = []
    for i, layer_idx in enumerate(gate_stats['layer_indices']):
        row = [
            iteration,
            layer_idx,
            float(gate_stats['bias_mean'][i]),
            float(gate_stats['bias_std'][i]),
            float(gate_stats['bias_min'][i]),
            float(gate_stats['bias_max'][i]),
            float(gate_stats['weight_mean'][i]),
            float(gate_stats['weight_std'][i]),
            float(gate_stats['weight_norm'][i]),
            float(gate_stats['gate_score_mean'][i]),
        ]
        rows.append(row)
    
   
    table = swanlab.echarts.Table()
    table.add(headers, rows)
    
    return table


def collect_ffn_gate_statistics(model: PreTrainedModel,
                                ds_engine=None) -> Dict[str, Any]:
    
    stats: Dict[str, Any] = {
        'layer_indices': [],
        'bias':          [],
        'gate_est':      [],
        'weight_norm':   [],
    }

    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return stats

    fp32_map: Dict[int, torch.Tensor] = {}
    if ds_engine is not None:
        try:
            optim = ds_engine.optimizer
            if hasattr(optim, 'fp32_groups'):
                for group in optim.fp32_groups:
                    for fp32_p in group:
                        if hasattr(fp32_p, '_param_id'):
                            fp32_map[fp32_p._param_id] = fp32_p
        except Exception:
            pass

    for layer_idx, layer in enumerate(model.model.layers):
        if not getattr(layer, 'ffn_output_gate', False):
            continue
        if not hasattr(layer, 'ffn_gate'):
            continue

        gate_mod = layer.ffn_gate
        bias_param = getattr(gate_mod, 'bias', None)
        if bias_param is None:
            continue

        bias_tensor = None
        if fp32_map:
            bias_id = getattr(bias_param, '_param_id', None)
            if bias_id is not None and bias_id in fp32_map:
                bias_tensor = fp32_map[bias_id].detach().float().cpu()
        if bias_tensor is None:
            bias_tensor = bias_param.detach().float().cpu()

        if bias_tensor.ndim == 0:
            bias_tensor = bias_tensor.unsqueeze(0)

        bias_val = float(bias_tensor.mean().item())

        # Aggregate weight norm across all gate parameters
        total_norm_sq = 0.0
        for p in gate_mod.parameters():
            total_norm_sq += p.detach().float().cpu().norm().item() ** 2
        weight_norm = total_norm_sq ** 0.5

        cfg = getattr(model, 'config', None)
        cfg_temp = getattr(cfg, 'ffn_gate_temperature', None) if cfg is not None else None
        if cfg_temp is not None:
            try:
                gate_temp = float(cfg_temp)
            except (TypeError, ValueError):
                gate_temp = 1.0 if bias_tensor.numel() == 1 else 8.0
        else:
            gate_temp = 1.0 if bias_tensor.numel() == 1 else 8.0

        gate_est = float((2.0 * torch.sigmoid(bias_tensor / gate_temp)).mean().item())

        stats['layer_indices'].append(layer_idx)
        stats['bias'].append(bias_val)
        stats['gate_est'].append(gate_est)
        stats['weight_norm'].append(weight_norm)

    return stats


def print_ffn_gate_console(
    gate_act: Dict[str, Any],
    iteration: int,
) -> None:
    
    if not gate_act['layer_indices']:
        print("⚠️  [FFN Gate] no ffn_gate in models（please check if ffn_output_gate=True）")
        return

    is_vector = gate_act.get('is_vector_gate', [False])[0]
    gate_type = "Vector-Gate" if is_vector else "Scalar-Gate"
    gate_temp_display = "8" if is_vector else "1"

    if is_vector:
        SEP = "═" * 86
        print(f"\n{SEP}")
        print(f"  Iter {iteration:5d} │ FFN {gate_type} (sigmoid, T={gate_temp_display})")
        print(SEP)
        print(f"  {'Layer':>5} │ {'mean':>11} │ {'std':>9} │ {'min':>9} │ {'max':>9} │ {'dim_std':>9} │ {'act%':>6}")
        print("─" * 86)
    else:
        SEP = "═" * 60
        print(f"\n{SEP}")
        print(f"  Iter {iteration:5d} │ FFN {gate_type} (sigmoid, T={gate_temp_display})")
        print(SEP)
        print(f"  {'Layer':>5} │ {'mean':>11} │ {'std':>9} │ {'min':>9} │ {'max':>9}")
        print("─" * 60)

    for i, layer_idx in enumerate(gate_act['layer_indices']):
        g_mean = gate_act['gate_mean'][i]
        g_std  = gate_act['gate_std'][i]
        g_min  = gate_act['gate_min'][i]
        g_max  = gate_act['gate_max'][i]

        if not np.isfinite(g_mean):
            if is_vector:
                print(f"  {layer_idx:>5} │ {'NaN':>11} │ {'NaN':>9} │ {'NaN':>9} │ {'NaN':>9} │ {'NaN':>9} │ {'NaN':>6}  ⚠️")
            else:
                print(f"  {layer_idx:>5} │ {'NaN':>11} │ {'NaN':>9} │ {'NaN':>9} │ {'NaN':>9}  ⚠️")
            continue

        if is_vector:
            dim_std = gate_act['gate_dim_std'][i]
            act_ratio = gate_act['gate_active_ratio'][i] * 100
            print(f"  {layer_idx:>5} │ {g_mean:>11.8f} │ {g_std:>9.6f} │ {g_min:>9.6f} │ {g_max:>9.6f} │ {dim_std:>9.6f} │ {act_ratio:>5.1f}%")
        else:
            print(f"  {layer_idx:>5} │ {g_mean:>11.8f} │ {g_std:>9.6f} │ {g_min:>9.6f} │ {g_max:>9.6f}")

    finite_means = [v for v in gate_act['gate_mean'] if np.isfinite(v)]
    nan_count    = len(gate_act['gate_mean']) - len(finite_means)
    sep_width = 86 if is_vector else 60
    print("─" * sep_width)
    if finite_means:
        avg  = float(np.mean(finite_means))
        lo   = float(np.min(gate_act['gate_min']))
        hi   = float(np.max(gate_act['gate_max']))
        spread = hi - lo
        warn = f"  ⚠️ {nan_count} NaN layers!" if nan_count > 0 else ""
        summary = f"  {'AVG':>5} │ {avg:>11.8f} │ range=[{lo:.6f}, {hi:.6f}] spread={spread:.6f}"
        if is_vector:
            avg_dim_std = float(np.mean(gate_act['gate_dim_std']))
            avg_act = float(np.mean(gate_act['gate_active_ratio'])) * 100
            summary += f" dim_std={avg_dim_std:.6f} act={avg_act:.1f}%"
        print(f"{summary}{warn}")
    else:
        print(f"  {'AVG':>5} │ {'ALL NaN':>11} │ ⚠️ NaN！")
    print(f"{'═' * sep_width}\n")


def create_ffn_gate_table(
    gate_act: Dict[str, Any],
    iteration: int,
) -> swanlab.echarts.Table:
   
    table = swanlab.echarts.Table()

    if not gate_act['layer_indices']:
        table.add(['iteration', 'layer', 'info'], [[iteration, 0, 'No FFN gates found']])
        return table

    is_vector = gate_act.get('is_vector_gate', [False])[0]

    headers = ['iteration', 'layer', 'gate_mean', 'gate_std', 'gate_min', 'gate_max']
    if is_vector:
        headers += ['gate_dim_std', 'gate_active_ratio']

    rows = []
    for i in range(len(gate_act['layer_indices'])):
        row = [
            iteration,
            gate_act['layer_indices'][i],
            round(gate_act['gate_mean'][i], 6),
            round(gate_act['gate_std'][i], 6),
            round(gate_act['gate_min'][i], 6),
            round(gate_act['gate_max'][i], 6),
        ]
        if is_vector:
            row.append(round(gate_act['gate_dim_std'][i], 6))
            row.append(round(gate_act['gate_active_ratio'][i], 4))
        rows.append(row)

    table.add(headers, rows)
    return table


def create_gate_activation_table(
    con_r_mean: float,
    con_r_std: float,
    par_r_mean: float,
    par_r_std: float,
    difference: float,
    sample_size: int,
    iteration: int
) -> swanlab.echarts.Table:
   
   
    headers = [
        'iteration',
        'task_type',
        'mean',
        'std',
        'difference',
        'sample_size'
    ]
    
    rows = [
        [
            iteration,
            'ConR (flag=True)',
            float(con_r_mean),
            float(con_r_std),
            float(difference),
            int(sample_size)
        ],
        [
            iteration,
            'ParR (flag=False)',
            float(par_r_mean),
            float(par_r_std),
            float(difference),
            int(sample_size)
        ]
    ]
    
   
    table = swanlab.echarts.Table()
    table.add(headers, rows)
    
    return table
