import argparse
import contextlib
import gc
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class GRPODatasetPadLeft(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        prompt_length: int = 512,
        answer_length: int = 1280,
        response_only: bool = True,
        use_template: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.answer_length = answer_length
        self.response_only = response_only
        self.use_template = use_template
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def __len__(self) -> int:
        return len(self.data)

    def _left_pad(self, token_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
        token_ids = token_ids[:max_length]
        pad_len = max_length - len(token_ids)
        return [self.pad_token_id] * pad_len + token_ids, [0] * pad_len + [1] * len(token_ids)

    def _right_pad(self, token_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
        token_ids = token_ids[:max_length]
        pad_len = max_length - len(token_ids)
        return token_ids + [self.pad_token_id] * pad_len, [1] * len(token_ids) + [0] * pad_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.response_only:
            question = item["Question"]
            answer = item["Answer"]
            if self.use_template:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = question
        else:
            prompt = item["text"]
            answer = ""

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.prompt_length,
            add_special_tokens=True,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.answer_length,
            add_special_tokens=False,
        )["input_ids"]

        prompt_ids, prompt_mask = self._left_pad(prompt_ids, self.prompt_length)
        answer_ids, answer_mask = self._right_pad(answer_ids, self.answer_length)

        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        attention_mask = torch.tensor(prompt_mask + answer_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "indices": torch.tensor(idx, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized forward-valuation LLM selection. "
            "Computes a sparse validation mean once, then scores train batches directly."
        )
    )
    parser.add_argument("--model_path", required=True, type=str, help="Path or HF name for the model")
    parser.add_argument(
        "--load_path",
        default=None,
        type=str,
        help="Optional PEFT adapter path to load on top of the base model.",
    )
    parser.add_argument("--data_path", required=True, type=str, help="Path or HF name for the dataset")
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help="Load dataset from disk instead of from HuggingFace",
    )
    parser.add_argument("--output_path", default="result_dict_llm_select_optimized.json", type=str)
    parser.add_argument("--batch_size_train", default=15, type=int)
    parser.add_argument("--batch_size_test", default=10, type=int)
    parser.add_argument("--prompt_length", default=512, type=int)
    parser.add_argument("--answer_length", default=1280, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--topk_abs", default=3, type=int, help="Per-position top-k |pb_diff| for union reduction")
    parser.add_argument(
        "--vocab_mode",
        default="total_unique",
        choices=["total_unique", "topk_unique"],
        help=(
            "Vocabulary mode. total_unique collects one global input-token vocabulary; "
            "topk_unique builds each batch vocabulary from per-position top-k predictions."
        ),
    )
    parser.add_argument(
        "--prediction_topk",
        default=32,
        type=int,
        help="Per-position prediction top-k used when vocab_mode=topk_unique.",
    )
    parser.add_argument(
        "--lowest_likelihood_ratio",
        default=1.0,
        type=float,
        help="Only keep the lowest-likelihood response positions when building representations.",
    )
    parser.add_argument("--select_ratio", default=0.1, type=float, help="Top ratio of train samples to select")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--score_device", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--embsum_batch_size",
        default=4,
        type=int,
        help="Compatibility arg from the old script. Not used in the optimized path.",
    )
    parser.add_argument(
        "--embsum_len_batch",
        default=100,
        type=int,
        help="Compatibility arg from the old script. Not used in the optimized path.",
    )
    return parser.parse_args()


def get_dataset(data_path: str, from_disk: bool = False):
    return load_from_disk(data_path) if from_disk else load_dataset(data_path)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def collect_unique_tokens(dataloaders: Iterable[DataLoader]) -> torch.Tensor:
    unique_token_ids = set()
    for dataloader in dataloaders:
        for batch in dataloader:
            unique_token_ids.update(batch["input_ids"].unique().tolist())
    return torch.tensor(sorted(unique_token_ids), dtype=torch.int64)


def forward_logits_and_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = base_model.lm_head(hidden)
        return logits, hidden

    if hasattr(model, "model") and hasattr(model, "lm_head"):
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = model.lm_head(hidden)
        return logits, hidden

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs.logits, outputs.hidden_states[-1]


def reduce_pbdiff_topk_union(
    pbdiff: torch.Tensor,
    candidate_vocab_ids: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pbdiff.shape[-1] == 0:
        return pbdiff, candidate_vocab_ids[:0]

    topk = min(topk, pbdiff.shape[-1])
    if topk == pbdiff.shape[-1]:
        return pbdiff, candidate_vocab_ids

    topk_idx = torch.topk(pbdiff.abs(), k=topk, dim=-1).indices
    union_idx = torch.unique(topk_idx.reshape(-1), sorted=True)
    return pbdiff.index_select(dim=-1, index=union_idx), candidate_vocab_ids.index_select(0, union_idx)


def compute_crop_length(attention_mask: torch.Tensor, prompt_length: int) -> int:
    response_lengths = attention_mask[:, prompt_length:].sum(dim=1)
    max_response_len = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
    if max_response_len <= 0:
        return prompt_length
    return min(attention_mask.shape[1], prompt_length + max_response_len + 1)


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        return int(model.config.text_config.hidden_size)
    raise ValueError("Could not infer hidden size from model config.")


def build_low_likelihood_position_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    lowest_likelihood_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
    if lowest_likelihood_ratio >= 1.0:
        return valid_mask, log_denom

    label_logits = logits.gather(-1, labels.unsqueeze(-1))
    label_probs = torch.exp(label_logits - log_denom).squeeze(-1)
    label_probs = torch.where(valid_mask, label_probs, torch.full_like(label_probs, float("inf")))
    valid_counts = valid_mask.sum(dim=1)
    keep_counts = torch.ceil(valid_counts.to(torch.float32) * lowest_likelihood_ratio).to(torch.long)
    keep_counts = torch.where(valid_counts > 0, keep_counts.clamp(min=1), keep_counts)
    order = torch.argsort(label_probs, dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(label_probs.shape[1], device=label_probs.device).unsqueeze(0).expand_as(order),
    )
    keep_mask = (ranks < keep_counts.unsqueeze(1)) & valid_mask
    return keep_mask, log_denom


def build_batch_vocabulary(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    prediction_topk: int,
) -> torch.Tensor:
    kept_logits = logits[valid_mask]
    if kept_logits.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=logits.device)

    topk_width = min(prediction_topk, logits.shape[-1])
    topk_ids = torch.topk(kept_logits, k=topk_width, dim=-1).indices
    return torch.unique(topk_ids.reshape(-1), sorted=True)


def map_vocab_to_global_positions(
    global_vocab_ids: torch.Tensor,
    subset_vocab_ids: torch.Tensor,
) -> torch.Tensor:
    positions = torch.searchsorted(global_vocab_ids, subset_vocab_ids)
    if positions.numel() == 0:
        return positions
    if positions.max().item() >= global_vocab_ids.numel():
        raise ValueError("subset_vocab_ids contains ids outside global_vocab_ids.")
    if not torch.equal(global_vocab_ids.index_select(0, positions), subset_vocab_ids):
        raise ValueError("subset_vocab_ids is not a subset of global_vocab_ids.")
    return positions


@torch.no_grad()
def compute_batch_representation(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    device: str,
    prompt_length: int,
    candidate_vocab_ids: Optional[torch.Tensor],
    topk_abs: int,
    lowest_likelihood_ratio: float,
    vocab_mode: str,
    prediction_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    crop_len = compute_crop_length(attention_mask, prompt_length)
    hidden_size = get_hidden_size(model)

    if crop_len <= prompt_length:
        empty = torch.zeros(
            (input_ids.shape[0], 0, hidden_size),
            dtype=torch.bfloat16,
        )
        empty_vocab = torch.empty(0, dtype=torch.int64)
        return empty, empty_vocab

    input_ids = input_ids[:, :crop_len]
    attention_mask = attention_mask[:, :crop_len]
    labels = input_ids[:, prompt_length:crop_len]
    valid_mask = attention_mask[:, prompt_length:crop_len].bool() # crop_len - prompt_length

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with amp_ctx:
        logits, hidden = forward_logits_and_hidden(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = logits[:, prompt_length - 1 : -1, :].float() 
    hidden = hidden[:, prompt_length - 1 : -1, :].float()

    if logits.shape[1] == 0:
        empty = torch.zeros(
            (input_ids.shape[0], 0, hidden_size),
            dtype=torch.bfloat16,
        )
        empty_vocab = torch.empty(0, dtype=torch.int64)
        return empty, empty_vocab

    keep_position_mask, log_denom = build_low_likelihood_position_mask(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        lowest_likelihood_ratio=lowest_likelihood_ratio,
    )  # length crop_len - prompt_length
    keep_counts = keep_position_mask.sum(dim=1)
    max_keep = int(keep_counts.max().item()) if keep_counts.numel() > 0 else 0
    if max_keep == 0:
        empty = torch.zeros(
            (input_ids.shape[0], 0, hidden_size),
            dtype=torch.bfloat16,
        )
        empty_vocab = torch.empty(0, dtype=torch.int64)
        return empty, empty_vocab

    if vocab_mode == "total_unique":
        if candidate_vocab_ids is None:
            raise ValueError("candidate_vocab_ids is required when vocab_mode='total_unique'.")
        batch_vocab_ids = candidate_vocab_ids
    else:
        batch_vocab_ids = build_batch_vocabulary(
            logits=logits,
            valid_mask=keep_position_mask,
            prediction_topk=prediction_topk,
        )

    if batch_vocab_ids.numel() == 0:
        empty = torch.zeros(
            (input_ids.shape[0], 0, hidden_size),
            dtype=torch.bfloat16,
        )
        return empty, batch_vocab_ids.detach().cpu()

    keep_order = torch.argsort((~keep_position_mask).to(torch.int64), dim=1)[:, :max_keep]
    kept_token_mask = (
        torch.arange(max_keep, device=logits.device).unsqueeze(0) < keep_counts.unsqueeze(1)
    )

    kept_hidden = hidden.gather(
        1,
        keep_order.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
    )
    kept_hidden = torch.nan_to_num(kept_hidden, nan=0.0, posinf=0.0, neginf=0.0)
    kept_hidden = kept_hidden * kept_token_mask.unsqueeze(-1).to(kept_hidden.dtype)

    selected_logits = logits.index_select(dim=-1, index=batch_vocab_ids)
    kept_selected_logits = selected_logits.gather(
        1,
        keep_order.unsqueeze(-1).expand(-1, -1, batch_vocab_ids.numel()),
    )
    kept_log_denom = log_denom.gather(1, keep_order.unsqueeze(-1))
    kept_labels = labels.gather(1, keep_order)
    prob_u = torch.exp(kept_selected_logits - kept_log_denom)
    one_hot_u = (kept_labels.unsqueeze(-1) == batch_vocab_ids.view(1, 1, -1)).to(prob_u.dtype)
    pb_diff = torch.nan_to_num(one_hot_u - prob_u, nan=0.0, posinf=0.0, neginf=0.0)
    pb_diff = pb_diff * kept_token_mask.unsqueeze(-1).to(pb_diff.dtype)

    pb_diff, reduced_vocab_ids = reduce_pbdiff_topk_union(
        pbdiff=pb_diff,
        candidate_vocab_ids=batch_vocab_ids,
        topk=topk_abs,
    )
    if reduced_vocab_ids.numel() == 0:
        empty = torch.zeros(
            (input_ids.shape[0], 0, hidden_size),
            dtype=torch.bfloat16,
        )
        return empty, reduced_vocab_ids.detach().cpu()

    proposed = torch.matmul(pb_diff.transpose(1, 2), kept_hidden)
    proposed = torch.nan_to_num(proposed, nan=0.0, posinf=0.0, neginf=0.0)
    return proposed.detach().cpu().to(torch.bfloat16), reduced_vocab_ids.detach().cpu()


def accumulate_validation_mean(
    dataloader_val: DataLoader,
    model: AutoModelForCausalLM,
    device: str,
    global_vocab_ids: torch.Tensor,
    prompt_length: int,
    topk_abs: int,
    lowest_likelihood_ratio: float,
) -> torch.Tensor:
    global_vocab_ids_device = global_vocab_ids.to(device, non_blocking=True)
    hidden_size = get_hidden_size(model)
    val_sum = torch.zeros((global_vocab_ids.numel(), hidden_size), dtype=torch.float32)
    num_val = 0

    for batch in tqdm(dataloader_val, desc="Building validation mean"):
        val_repr, vocab_ids = compute_batch_representation(
            model=model,
            batch=batch,
            device=device,
            prompt_length=prompt_length,
            candidate_vocab_ids=global_vocab_ids_device,
            topk_abs=topk_abs,
            lowest_likelihood_ratio=lowest_likelihood_ratio,
            vocab_mode="total_unique",
            prediction_topk=0,
        )
        if vocab_ids.numel() > 0:
            positions = map_vocab_to_global_positions(global_vocab_ids, vocab_ids)
            val_sum.index_add_(0, positions, val_repr.float().sum(dim=0))
        num_val += int(batch["input_ids"].shape[0])

    if num_val == 0:
        raise ValueError("Validation set is empty.")
    return val_sum / float(num_val)


def collect_sparse_validation_references(
    dataloader_val: DataLoader,
    model: AutoModelForCausalLM,
    device: str,
    prompt_length: int,
    topk_abs: int,
    lowest_likelihood_ratio: float,
    prediction_topk: int,
) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    validation_refs = []
    num_val = 0

    for batch in tqdm(dataloader_val, desc="Building sparse validation refs"):
        val_repr, vocab_ids = compute_batch_representation(
            model=model,
            batch=batch,
            device=device,
            prompt_length=prompt_length,
            candidate_vocab_ids=None,
            topk_abs=topk_abs,
            lowest_likelihood_ratio=lowest_likelihood_ratio,
            vocab_mode="topk_unique",
            prediction_topk=prediction_topk,
        )
        if vocab_ids.numel() > 0:
            validation_refs.append(
                {
                    "vocab_ids": vocab_ids,
                    "repr_sum": val_repr.sum(dim=0).to(torch.bfloat16),
                }
            )
        num_val += int(batch["input_ids"].shape[0])

    if num_val == 0:
        raise ValueError("Validation set is empty.")
    return validation_refs, num_val


def merge_sparse_validation_refs(
    validation_refs: List[Dict[str, torch.Tensor]],
    num_val: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_val == 0:
        raise ValueError("Validation set is empty.")
    if not validation_refs:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.zeros((0, hidden_size), dtype=torch.float32),
        )

    merged_vocab_ids = torch.unique(
        torch.cat([ref["vocab_ids"] for ref in validation_refs], dim=0),
        sorted=True,
    )
    val_sum = torch.zeros((merged_vocab_ids.numel(), hidden_size), dtype=torch.float32)
    for val_ref in validation_refs:
        positions = map_vocab_to_global_positions(merged_vocab_ids, val_ref["vocab_ids"])
        val_sum.index_add_(0, positions, val_ref["repr_sum"].float())
    return merged_vocab_ids, val_sum / float(num_val)


def get_vocab_intersection_indices(
    query_vocab_ids: torch.Tensor,
    reference_vocab_ids: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if query_vocab_ids.numel() == 0 or reference_vocab_ids.numel() == 0:
        return None, None

    positions = torch.searchsorted(reference_vocab_ids, query_vocab_ids)
    valid = positions < reference_vocab_ids.numel()
    if not valid.any():
        return None, None

    query_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    reference_idx = positions[valid]
    matched = reference_vocab_ids[reference_idx] == query_vocab_ids[valid]
    if not matched.any():
        return None, None

    return query_idx[matched], reference_idx[matched]


def score_against_validation_mean(
    train_repr: torch.Tensor,
    train_vocab_ids: torch.Tensor,
    validation_vocab_ids: torch.Tensor,
    val_mean: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    batch_scores = torch.zeros(train_repr.shape[0], dtype=torch.float32)
    if train_vocab_ids.numel() == 0 or validation_vocab_ids.numel() == 0:
        return batch_scores

    train_idx, val_idx = get_vocab_intersection_indices(
        query_vocab_ids=train_vocab_ids,
        reference_vocab_ids=validation_vocab_ids,
    )
    if train_idx is None or val_idx is None:
        return batch_scores

    train_common = train_repr[:, train_idx, :].to(
        score_device,
        dtype=torch.float32,
        non_blocking=True,
    )
    val_common = val_mean.index_select(0, val_idx).to(
        score_device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return torch.nan_to_num(
        (train_common * val_common.unsqueeze(0)).sum(dim=(1, 2)),
        nan=0.0,
        posinf=1e30,
        neginf=-1e30,
    ).cpu()


def score_train_samples(
    dataloader_train: DataLoader,
    model: AutoModelForCausalLM,
    device: str,
    score_device: str,
    validation_vocab_ids: Optional[torch.Tensor],
    val_mean: Optional[torch.Tensor],
    prompt_length: int,
    topk_abs: int,
    lowest_likelihood_ratio: float,
    vocab_mode: str,
    prediction_topk: int,
) -> torch.Tensor:
    scores = []

    for batch in tqdm(dataloader_train, desc="Scoring train samples"):
        batch_vocab_ids = (
            torch.unique(batch["input_ids"], sorted=True).to(device, non_blocking=True)
            if vocab_mode == "total_unique"
            else None
        )
        train_repr, reduced_vocab_ids = compute_batch_representation(
            model=model,
            batch=batch,
            device=device,
            prompt_length=prompt_length,
            candidate_vocab_ids=batch_vocab_ids,
            topk_abs=topk_abs,
            lowest_likelihood_ratio=lowest_likelihood_ratio,
            vocab_mode=vocab_mode,
            prediction_topk=prediction_topk,
        )

        if reduced_vocab_ids.numel() == 0:
            batch_scores = torch.zeros(train_repr.shape[0], dtype=torch.float32)
        else:
            if validation_vocab_ids is None or val_mean is None:
                raise ValueError("validation_vocab_ids and val_mean are required for scoring.")
            batch_scores = score_against_validation_mean(
                train_repr=train_repr,
                train_vocab_ids=reduced_vocab_ids,
                validation_vocab_ids=validation_vocab_ids,
                val_mean=val_mean,
                score_device=score_device,
            )
        scores.append(batch_scores)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(scores, dim=0)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.lowest_likelihood_ratio <= 1.0):
        raise ValueError(
            f"lowest_likelihood_ratio must be in (0, 1], got {args.lowest_likelihood_ratio}."
        )
    if args.vocab_mode == "topk_unique" and args.prediction_topk <= 0:
        raise ValueError(f"prediction_topk must be positive, got {args.prediction_topk}.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    score_device = args.score_device or args.device

    dataset = get_dataset(args.data_path, args.load_from_disk)
    data_train = dataset["train"]
    data_val = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset_train = GRPODatasetPadLeft(
        data_train,
        tokenizer=tokenizer,
        prompt_length=args.prompt_length,
        answer_length=args.answer_length,
        response_only=True,
        use_template=False,
    )
    dataset_val = GRPODatasetPadLeft(
        data_val,
        tokenizer=tokenizer,
        prompt_length=args.prompt_length,
        answer_length=args.answer_length,
        response_only=True,
        use_template=False,
    )
    dataloader_train = build_dataloader(
        dataset=dataset_train,
        batch_size=args.batch_size_train,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor,
    )
    dataloader_val = build_dataloader(
        dataset=dataset_val,
        batch_size=args.batch_size_test,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor,
    )

    start_time = time.time()
    global_vocab_ids = None
    validation_vocab_ids = None
    if args.vocab_mode == "total_unique":
        print("Collecting unique tokens...")
        global_vocab_ids = collect_unique_tokens([dataloader_train, dataloader_val])
        validation_vocab_ids = global_vocab_ids
        print(f"Unique tokens: {global_vocab_ids.numel()}")
    else:
        print(
            f"Using vocab_mode=topk_unique with prediction_topk={args.prediction_topk}. "
            "No global unique-token collection."
        )
    print(f"lowest_likelihood_ratio: {args.lowest_likelihood_ratio}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    if args.load_path is not None:
        model = PeftModel.from_pretrained(model, args.load_path)  # finetune the model slightly to meet the unconstrained requirement.
    model.to(args.device)
    model.eval()

    val_start = time.time()
    val_mean = None
    validation_ref_batches = None
    num_val = None
    validation_vocab_sizes = []
    if args.vocab_mode == "total_unique":
        if global_vocab_ids is None:
            raise ValueError("global_vocab_ids is required when vocab_mode='total_unique'.")
        val_mean = accumulate_validation_mean(
            dataloader_val=dataloader_val,
            model=model,
            device=args.device,
            global_vocab_ids=global_vocab_ids,
            prompt_length=args.prompt_length,
            topk_abs=args.topk_abs,
            lowest_likelihood_ratio=1.0,
        )
    else:
        validation_refs, num_val = collect_sparse_validation_references(
            dataloader_val=dataloader_val,
            model=model,
            device=args.device,
            prompt_length=args.prompt_length,
            topk_abs=args.topk_abs,
            lowest_likelihood_ratio=1.0,
            prediction_topk=args.prediction_topk,
        )
        validation_ref_batches = len(validation_refs)
        validation_vocab_sizes = [int(ref["vocab_ids"].numel()) for ref in validation_refs]
        validation_vocab_ids, val_mean = merge_sparse_validation_refs(
            validation_refs=validation_refs,
            num_val=num_val,
            hidden_size=get_hidden_size(model),
        )
    val_time = time.time() - val_start
    if val_mean is not None:
        print(f"Validation mean shape: {tuple(val_mean.shape)}")
    if validation_vocab_sizes:
        print(
            f"Validation refs: {validation_ref_batches}, "
            f"vocab_size[min/mean/max]="
            f"{min(validation_vocab_sizes)}/{np.mean(validation_vocab_sizes):.1f}/{max(validation_vocab_sizes)}"
        )
    elif val_mean is None:
        print("Validation refs: 0 non-empty vocabularies")

    score_start = time.time()
    train_scores = score_train_samples(
        dataloader_train=dataloader_train,
        model=model,
        device=args.device,
        score_device=score_device,
        validation_vocab_ids=validation_vocab_ids,
        val_mean=val_mean,
        prompt_length=args.prompt_length,
        topk_abs=args.topk_abs,
        lowest_likelihood_ratio=args.lowest_likelihood_ratio,
        vocab_mode=args.vocab_mode,
        prediction_topk=args.prediction_topk,
    )
    score_time = time.time() - score_start

    num_select = max(1, int(len(train_scores) * args.select_ratio))
    top_indices = torch.topk(train_scores, k=num_select).indices.cpu()
    selected_data = data_train.select(top_indices.tolist())

    result_dict = {
        "model_path": args.model_path,
        "load_path": args.load_path,
        "data_path": args.data_path,
        "vocab_mode": args.vocab_mode,
        "prediction_topk": int(args.prediction_topk),
        "unique_tokens": int(global_vocab_ids.numel()) if global_vocab_ids is not None else None,
        "validation_vocab_size": int(validation_vocab_ids.numel()) if validation_vocab_ids is not None else None,
        "val_mean_shape": [int(val_mean.shape[0]), int(val_mean.shape[1])] if val_mean is not None else None,
        "validation_ref_batches": validation_ref_batches,
        "validation_vocab_min": min(validation_vocab_sizes) if validation_vocab_sizes else None,
        "validation_vocab_mean": float(np.mean(validation_vocab_sizes)) if validation_vocab_sizes else None,
        "validation_vocab_max": max(validation_vocab_sizes) if validation_vocab_sizes else None,
        "topk_abs": int(args.topk_abs),
        "lowest_likelihood_ratio": float(args.lowest_likelihood_ratio),
        "select_ratio": float(args.select_ratio),
        "num_selected": int(num_select),
        "val_build_sec": float(val_time),
        "train_score_sec": float(score_time),
        "total_sec": float(time.time() - start_time),
    }
    if "clean_label" in selected_data.column_names:
        clean_ratio = float(np.mean(np.array(selected_data["clean_label"], dtype=np.float32)))
        result_dict["selected_clean_ratio"] = clean_ratio
        print(clean_ratio)
    else:
        print("clean_label not found in train set; skipping clean-ratio report.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
