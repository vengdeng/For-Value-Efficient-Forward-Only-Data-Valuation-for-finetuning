import argparse
import contextlib
import gc
import io
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


QUESTION_TEXT = "What is the animal in the image?"
ANSWER_PREFIX = "It is a "
LLAMA_PROMPT_PREFIX = "<|image|><|begin_of_text|>"
LLAMA_START_INDEX = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized VL forward valuation: compute sample-level representations "
            "directly and score them in chunks."
        )
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_path", type=str, default="dataset_train_noisy_0.6")
    parser.add_argument("--test_path", type=str, default="dataset_test_clean")
    parser.add_argument("--batch_size_train", type=int, default=30)
    parser.add_argument("--batch_size_test", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=800)
    parser.add_argument(
        "--vocab_mode",
        type=str,
        default="total_unique",
        choices=["total_unique", "topk_unique"],
        help="Vocabulary mode. total_unique uses all observed input tokens; topk_unique uses per-batch top-k predicted tokens.",
    )
    parser.add_argument(
        "--prediction_topk",
        type=int,
        default=32,
        help="Per-position prediction top-k used when vocab_mode=topk_unique.",
    )
    parser.add_argument("--mislabel_ratio", type=float, default=0.6)
    parser.add_argument("--assistant_token_id", type=int, default=77091)
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="If omitted, infer from the first train batch using assistant_token_id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--score_device", type=str, default=None)
    parser.add_argument("--score_chunk_size", type=int, default=128)
    parser.add_argument("--result_path", type=str, default="result_dict_vl_optimized.json")
    parser.add_argument("--time_path", type=str, default="time_dict_vl_optimized.json")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


class VLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.data[idx]
        image = item["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type at index {idx}: {type(image)}")
        return {"image": image.convert("RGB"), "text": item["text"]}


def processor_has_chat_template(processor) -> bool:
    return bool(getattr(processor, "chat_template", None))


def build_collate_fn(processor, image_size: int, max_length: int):
    has_chat_template = processor_has_chat_template(processor)

    def collate_fn(batch):
        resized_images = []
        prompts = []
        for item in batch:
            image = item["image"].resize((image_size, image_size))
            resized_images.append([image])
            if has_chat_template:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": QUESTION_TEXT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": ANSWER_PREFIX + item["text"]},
                        ],
                    },
                ]
                prompts.append(
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            else:
                prompts.append(
                    LLAMA_PROMPT_PREFIX + QUESTION_TEXT + ANSWER_PREFIX + item["text"]
                )

        return processor(
            text=prompts,
            images=resized_images,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )

    return collate_fn


def move_batch_to_device(batch, device: str):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def infer_start_index(input_ids: torch.Tensor, assistant_token_id: int) -> int:
    matches = (input_ids == assistant_token_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return LLAMA_START_INDEX
    row = matches[0, 0]
    row_matches = matches[matches[:, 0] == row]
    return int(row_matches[-1, 1].item())


def resolve_start_index(
    dataloader: DataLoader,
    start_index: Optional[int],
    assistant_token_id: int,
) -> int:
    if start_index is not None:
        return start_index

    for batch in dataloader:
        return infer_start_index(batch["input_ids"], assistant_token_id)

    raise ValueError("Failed to resolve start_index from the dataloader.")


def collect_total_unique_tokens_and_start_index(
    dataloaders: Iterable[DataLoader],
    start_index: Optional[int],
    assistant_token_id: int,
    target_vocab_only: bool = False,
) -> Tuple[torch.Tensor, int]:
    unique_token_ids = set()
    resolved_start_index = start_index

    for dataloader in dataloaders:
        for batch in dataloader:
            input_ids = batch["input_ids"]
            if resolved_start_index is None:
                resolved_start_index = infer_start_index(input_ids, assistant_token_id)
            if target_vocab_only:
                target_ids = input_ids[:, resolved_start_index + 1 :]
                unique_token_ids.update(target_ids.unique().tolist())
            else:
                unique_token_ids.update(input_ids.unique().tolist())

    if resolved_start_index is None:
        raise ValueError("Failed to resolve start_index.")

    return torch.tensor(sorted(unique_token_ids), dtype=torch.int64), resolved_start_index


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


@torch.no_grad()
def compute_batch_representation(
    model: AutoModelForImageTextToText,
    batch,
    device: str,
    start_index: int,
    vocab_mode: str,
    prediction_topk: int,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
    global_vocab_ids_device: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_device = move_batch_to_device(batch, device)

    use_amp = device.startswith("cuda")
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )
    with amp_ctx:
        outputs = model(**batch_device)

    logits = outputs.logits[:, start_index:-1].float()
    hidden = outputs.hidden_states[-1][:, start_index:-1].float()
    input_ids = batch_device["input_ids"]
    attention_mask = batch_device["attention_mask"]
    labels = input_ids[:, start_index + 1 :]
    valid_mask = attention_mask[:, start_index:-1].bool()
    mask = valid_mask.unsqueeze(-1).to(hidden.dtype)

    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None or global_vocab_ids_device is None:
            raise ValueError("global_vocab_ids are required when vocab_mode='total_unique'.")
        batch_vocab_ids = global_vocab_ids_device
        batch_vocab_ids_cpu = global_vocab_ids_cpu
    else:
        batch_vocab_ids = build_batch_vocabulary(
            logits=logits,
            valid_mask=valid_mask,
            prediction_topk=prediction_topk,
        )
        batch_vocab_ids_cpu = batch_vocab_ids.detach().cpu()

    if logits.shape[1] == 0 or batch_vocab_ids.numel() == 0:
        empty = hidden.new_zeros((input_ids.shape[0], batch_vocab_ids.numel(), hidden.shape[-1]))
        return empty.cpu().to(torch.bfloat16), batch_vocab_ids_cpu

    selected_logits = logits.index_select(dim=-1, index=batch_vocab_ids)
    log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
    prob_u = torch.exp(selected_logits - log_denom)
    one_hot_u = (labels.unsqueeze(-1) == batch_vocab_ids.view(1, 1, -1)).to(prob_u.dtype)
    pb_diff = torch.nan_to_num((one_hot_u - prob_u) * mask, nan=0.0, posinf=0.0, neginf=0.0)
    hidden = torch.nan_to_num(hidden * mask, nan=0.0, posinf=0.0, neginf=0.0)
    proposed = torch.bmm(pb_diff.transpose(1, 2), hidden)
    proposed = torch.nan_to_num(proposed, nan=0.0, posinf=0.0, neginf=0.0)
    return proposed.detach().cpu().to(torch.bfloat16), batch_vocab_ids_cpu


def extract_representations(
    dataloader: DataLoader,
    model: AutoModelForImageTextToText,
    device: str,
    start_index: int,
    vocab_mode: str,
    prediction_topk: int,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor | List[Dict[str, torch.Tensor]]:
    global_vocab_ids_device = None
    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None:
            raise ValueError("global_vocab_ids_cpu is required when vocab_mode='total_unique'.")
        global_vocab_ids_device = global_vocab_ids_cpu.to(device, non_blocking=True)
        representations: List[torch.Tensor] = []
    else:
        representations = []

    for step, batch in enumerate(dataloader):
        proposed, vocab_ids = compute_batch_representation(
            model=model,
            batch=batch,
            device=device,
            start_index=start_index,
            vocab_mode=vocab_mode,
            prediction_topk=prediction_topk,
            global_vocab_ids_cpu=global_vocab_ids_cpu,
            global_vocab_ids_device=global_vocab_ids_device,
        )
        if vocab_mode == "total_unique":
            representations.append(proposed)
        else:
            representations.append(
                {
                    "proposed": proposed,
                    "vocab_ids": vocab_ids,
                }
            )
        if (step + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if vocab_mode == "total_unique":
        return torch.cat(representations, dim=0)
    return representations


def dense_pairwise_dot(
    test_repr: torch.Tensor,
    train_repr: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    test_flat = test_repr.to(score_device, dtype=torch.float32, non_blocking=True).reshape(test_repr.shape[0], -1)
    train_flat = train_repr.to(score_device, dtype=torch.float32, non_blocking=True).reshape(train_repr.shape[0], -1)
    sim = torch.matmul(test_flat, train_flat.transpose(0, 1))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e30, neginf=-1e30)
    return sim.cpu()


def chunked_pairwise_dot(
    test_repr: torch.Tensor,
    train_repr: torch.Tensor,
    score_device: str,
    chunk_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, train_repr.shape[0], chunk_size):
        end = min(start + chunk_size, train_repr.shape[0])
        rows.append(
            dense_pairwise_dot(
                test_repr=test_repr,
                train_repr=train_repr[start:end],
                score_device=score_device,
            )
        )
    return torch.cat(rows, dim=1)


def get_vocab_intersection_indices(
    test_vocab_ids: torch.Tensor,
    train_vocab_ids: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if test_vocab_ids.numel() == 0 or train_vocab_ids.numel() == 0:
        return None, None

    positions = torch.searchsorted(train_vocab_ids, test_vocab_ids)
    valid = positions < train_vocab_ids.numel()
    if not valid.any():
        return None, None

    test_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    train_idx = positions[valid]
    matched = train_vocab_ids[train_idx] == test_vocab_ids[valid]
    if not matched.any():
        return None, None

    return test_idx[matched], train_idx[matched]


def aligned_vector_pairwise_dot(
    test_values: torch.Tensor,
    test_vocab_ids: torch.Tensor,
    train_values: torch.Tensor,
    train_vocab_ids: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    test_idx, train_idx = get_vocab_intersection_indices(test_vocab_ids, train_vocab_ids)
    if test_idx is None or train_idx is None or test_idx.numel() == 0:
        return torch.zeros((test_values.shape[0], train_values.shape[0]), dtype=torch.float32)

    test_common = test_values[:, test_idx, :].to(score_device, dtype=torch.float32, non_blocking=True)
    train_common = train_values[:, train_idx, :].to(score_device, dtype=torch.float32, non_blocking=True)
    test_flat = test_common.reshape(test_common.shape[0], -1)
    train_flat = train_common.reshape(train_common.shape[0], -1)
    sim = torch.matmul(test_flat, train_flat.transpose(0, 1))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e30, neginf=-1e30)
    return sim.cpu()


def score_test_against_train(
    dataloader_test: DataLoader,
    model: AutoModelForImageTextToText,
    device: str,
    score_device: str,
    start_index: int,
    train_repr: torch.Tensor | List[Dict[str, torch.Tensor]],
    chunk_size: int,
    vocab_mode: str,
    prediction_topk: int,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score_rows = []
    global_vocab_ids_device = None
    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None:
            raise ValueError("global_vocab_ids_cpu is required when vocab_mode='total_unique'.")
        global_vocab_ids_device = global_vocab_ids_cpu.to(device, non_blocking=True)

    for step, batch in enumerate(dataloader_test):
        test_repr, test_vocab_ids = compute_batch_representation(
            model=model,
            batch=batch,
            device=device,
            start_index=start_index,
            vocab_mode=vocab_mode,
            prediction_topk=prediction_topk,
            global_vocab_ids_cpu=global_vocab_ids_cpu,
            global_vocab_ids_device=global_vocab_ids_device,
        )
        if vocab_mode == "total_unique":
            if not isinstance(train_repr, torch.Tensor):
                raise ValueError("train_repr must be a tensor when vocab_mode='total_unique'.")
            score_rows.append(
                chunked_pairwise_dot(
                    test_repr=test_repr,
                    train_repr=train_repr,
                    score_device=score_device,
                    chunk_size=chunk_size,
                )
            )
        else:
            if not isinstance(train_repr, list):
                raise ValueError("train_repr must be a list when vocab_mode='topk_unique'.")
            score_parts = []
            for train_batch_repr in train_repr:
                score_parts.append(
                    aligned_vector_pairwise_dot(
                        test_values=test_repr,
                        test_vocab_ids=test_vocab_ids,
                        train_values=train_batch_repr["proposed"],
                        train_vocab_ids=train_batch_repr["vocab_ids"],
                        score_device=score_device,
                    )
                )
            score_rows.append(torch.cat(score_parts, dim=1))
        if (step + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(score_rows, dim=0)


def evaluate_metrics(
    values: torch.Tensor,
    data_train,
    data_test,
    mislabel_ratio: float,
) -> Dict[str, float]:
    proposed_auc_list = []
    proposed_recall_list = []

    for i in range(len(data_test["text"])):
        gt_label = data_test["text"][i]
        gt_array = np.array(
            [
                1 if (tr_label == gt_label) and (noise_label == 0) else 0
                for tr_label, noise_label in zip(data_train["text"], data_train["label"])
            ]
        )

        probabilities = values[i].cpu().float().numpy()
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1e30, neginf=-1e30)
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        proposed_auc_list.append(float(roc_auc_score(gt_array, probabilities)))

    val_array = np.array(data_test["text"])
    for i in range(len(data_test["text"])):
        gt_label = data_test["text"][i]
        n_label = int(np.sum(val_array == gt_label) * mislabel_ratio)
        if n_label <= 0:
            continue
        sorted_index = np.argsort(values[i].cpu().float().numpy())[::-1]
        sorted_array = np.array([data_train["label"][j] for j in sorted_index])
        proposed_recall_list.append(float(np.count_nonzero(sorted_array[:n_label] == 0) / n_label))

    return {
        "proposed_auc": float(np.mean(proposed_auc_list)),
        "proposed_auc_std": float(np.std(proposed_auc_list)),
        "proposed_recall": float(np.mean(proposed_recall_list)),
        "proposed_recall_std": float(np.std(proposed_recall_list)),
    }


def main() -> None:
    args = parse_args()
    score_device = args.score_device or args.device
    start_time = time.time()
    if args.vocab_mode == "topk_unique" and args.prediction_topk <= 0:
        raise ValueError(f"prediction_topk must be positive, got {args.prediction_topk}.")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    collate_fn = build_collate_fn(
        processor=processor,
        image_size=args.image_size,
        max_length=args.max_length,
    )
    target_vocab_only = not processor_has_chat_template(processor)

    loader_kwargs = {
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    data_train = load_from_disk(args.train_path)
    data_test = load_from_disk(args.test_path)

    dataloader_train = DataLoader(
        VLDataset(data_train),
        batch_size=args.batch_size_train,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    dataloader_test = DataLoader(
        VLDataset(data_test),
        batch_size=args.batch_size_test,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    if args.vocab_mode == "total_unique":
        print("Collecting unique tokens and resolving start_index...")
        unq_tokens, start_index = collect_total_unique_tokens_and_start_index(
            dataloaders=[dataloader_train, dataloader_test],
            start_index=args.start_index,
            assistant_token_id=args.assistant_token_id,
            target_vocab_only=target_vocab_only,
        )
        unique_label = "Unique target tokens" if target_vocab_only else "Unique tokens"
        print(f"{unique_label}: {unq_tokens.numel()}")
    else:
        print(
            f"Using vocab_mode=topk_unique with prediction_topk={args.prediction_topk}. "
            "No global unique-token collection."
        )
        unq_tokens = None
        start_index = resolve_start_index(
            dataloader=dataloader_train,
            start_index=args.start_index,
            assistant_token_id=args.assistant_token_id,
        )
    print(f"start_index: {start_index}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        output_hidden_states=True,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()

    feat_start = time.time()
    print("Extracting train representations...")
    train_repr = extract_representations(
        dataloader=dataloader_train,
        model=model,
        device=args.device,
        start_index=start_index,
        vocab_mode=args.vocab_mode,
        prediction_topk=args.prediction_topk,
        global_vocab_ids_cpu=unq_tokens,
    )
    feat_time = time.time() - feat_start
    if isinstance(train_repr, torch.Tensor):
        print(f"Train representation shape: {tuple(train_repr.shape)}")
    else:
        vocab_sizes = [int(batch_repr["vocab_ids"].numel()) for batch_repr in train_repr]
        print(
            f"Train batches: {len(train_repr)}, "
            f"vocab_size[min/mean/max]={min(vocab_sizes)}/{np.mean(vocab_sizes):.1f}/{max(vocab_sizes)}"
        )

    score_start = time.time()
    print("Scoring test batches...")
    values = score_test_against_train(
        dataloader_test=dataloader_test,
        model=model,
        device=args.device,
        score_device=score_device,
        start_index=start_index,
        train_repr=train_repr,
        chunk_size=args.score_chunk_size,
        vocab_mode=args.vocab_mode,
        prediction_topk=args.prediction_topk,
        global_vocab_ids_cpu=unq_tokens,
    )
    score_time = time.time() - score_start

    metrics = evaluate_metrics(
        values=values,
        data_train=data_train,
        data_test=data_test,
        mislabel_ratio=args.mislabel_ratio,
    )
    total_time = time.time() - start_time

    print(f"proposed AUC: {metrics['proposed_auc']:.3f}/{metrics['proposed_auc_std']:.3f}")
    print(f"proposed Recall: {metrics['proposed_recall']:.3f}/{metrics['proposed_recall_std']:.3f}")
    print(
        f"Timing: total={total_time:.1f}s, train_feature={feat_time:.1f}s, "
        f"test_score={score_time:.1f}s"
    )

    result_dict = {
        "proposed_auc": metrics["proposed_auc"],
        "proposed_auc_std": metrics["proposed_auc_std"],
        "proposed_recall": metrics["proposed_recall"],
        "proposed_recall_std": metrics["proposed_recall_std"],
        "vocab_mode": args.vocab_mode,
        "prediction_topk": int(args.prediction_topk),
        "unique_tokens": int(unq_tokens.numel()) if unq_tokens is not None else None,
        "start_index": int(start_index),
    }
    time_dict = {
        args.model_name: total_time,
        "train_feature_sec": feat_time,
        "test_score_sec": score_time,
    }

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f)
    with open(args.time_path, "w", encoding="utf-8") as f:
        json.dump(time_dict, f)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
