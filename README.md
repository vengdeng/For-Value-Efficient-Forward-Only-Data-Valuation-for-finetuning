# ForwardValuation

Official implementation of the ACL 2026 paper **For-Value: Efficient Forward-Only Data Valuation for Finetuning LLMs and VLMs**.

For-Value estimates training-sample value for language-model and vision-language-model finetuning using forward-pass representations only. The method combines hidden states with prediction-probability differences (`pb_diff`) and scores each training sample against validation samples or a validation mean, avoiding expensive backward-pass or retraining-based valuation.

This repository focuses on optimized, streaming-style entry points for LLM and VLM valuation that avoid storing the full train-test tensor whenever possible.

## Entry Points

| File | Purpose |
| --- | --- |
| `forvalue_streaming.py` | Text forward valuation for class-style train/test datasets. Streams test scoring, supports global or batch-local vocabularies, and reports AUC/recall. |
| `forvalue_VL_optimized.py` | Vision-language forward valuation for image-text datasets. Computes batch representations, scores in chunks, and supports `total_unique` / `topk_unique` vocabulary modes. |
| `forvalue_LLM_select_optimized.py` | LLM data-selection pipeline. Builds a validation reference, scores train samples, selects the top `--select_ratio`, and optionally reports the selected clean-label ratio. |
| `utils.py` | Shared dataset and valuation helpers. |
| `noise_huatuo.py`, `dog_cat.py` | Dataset construction and preprocessing helpers. |
| `llm_select.sh` | Legacy example command. Prefer the commands below because the optimized script is now the maintained selection path. |

## Setup

Use an environment with PyTorch, Transformers, Datasets, scikit-learn, NumPy, tqdm, Pillow, and PEFT installed:

```bash
pip install torch transformers datasets scikit-learn numpy tqdm pillow peft
```

For GPU runs, install a PyTorch build that matches your CUDA version. If models and datasets are already cached locally, pass `--local_files_only` to avoid network access.

## Data Layout

Text valuation expects HuggingFace datasets saved on disk as:

```text
dataset/{dataset_name}_train.hf
dataset/{dataset_name}_test.hf
```

The default text dataset is:

```text
math_without_reason
```

Vision-language valuation expects disk datasets such as:

```text
dataset_train_noisy_0.6
dataset_test_clean
```

LLM selection expects a HuggingFace dataset or disk dataset with `train` and `test` splits and response-style columns:

```text
Question
Answer
```

If `clean_label` exists in the training split, `forvalue_LLM_select_optimized.py` reports `selected_clean_ratio` for the selected subset.

## Recommended Commands

Run commands from the repository root:

```bash
cd /data/dwenlong/ForwardValuation
```

### Text Forward Valuation

Global vocabulary mode:

```bash
python3 forvalue_streaming.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --dataset_name math_without_reason \
  --dataset_dir dataset \
  --embed_device cuda:0 \
  --score_device cuda:0 \
  --batch_size 50 \
  --vocab_mode total_unique
```

Batch-local top-k vocabulary mode:

```bash
python3 forvalue_streaming.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --dataset_name math_without_reason \
  --dataset_dir dataset \
  --embed_device cuda:0 \
  --score_device cuda:0 \
  --batch_size 50 \
  --vocab_mode topk_unique \
  --prediction_topk 5 \
  --lowest_likelihood_ratio 1.0
```

Use `--approximate_proposed` to approximate the proposed score with token-sum similarity multiplied by average-`pb_diff` similarity, skipping full proposed-representation scoring.

### Vision-Language Valuation

```bash
python3 forvalue_VL_optimized.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --train_path dataset_train_noisy_0.6 \
  --test_path dataset_test_clean \
  --device cuda:0 \
  --score_device cuda:0 \
  --batch_size_train 30 \
  --batch_size_test 30 \
  --vocab_mode total_unique
```

Use `--vocab_mode topk_unique --prediction_topk 5` when global vocabulary construction is too expensive.

### LLM Data Selection

```bash
python3 forvalue_LLM_select_optimized.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --data_path Medical_noise_split_new \
  --load_from_disk \
  --device cuda:0 \
  --score_device cuda:0 \
  --batch_size_train 15 \
  --batch_size_test 10 \
  --select_ratio 0.1 \
  --lowest_likelihood_ratio 0.2 \
  --vocab_mode topk_unique \
  --prediction_topk 5
```

If the base model is weak on the target domain, first fine-tune it briefly on the available data and save a PEFT/LoRA adapter. Then run selection with `--load_path`:

```bash
python3 forvalue_LLM_select_optimized.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --load_path /data/dwenlong/ForwardValuation/ckpts_split_1ep_8gpu_lora/sft_stage1/checkpoint-0-1313/tfmr \
  --data_path Medical_noise_split_new \
  --load_from_disk \
  --device cuda:1 \
  --score_device cuda:1 \
  --batch_size_train 20 \
  --batch_size_test 20 \
  --select_ratio 0.1 \
  --lowest_likelihood_ratio 1.0 \
  --vocab_mode topk_unique \
  --prediction_topk 5
```

## Key Options

| Option | Meaning |
| --- | --- |
| `--vocab_mode total_unique` | Build one global vocabulary from observed input tokens. This is more faithful but can be slower and use more memory. |
| `--vocab_mode topk_unique` | Build batch-local vocabularies from each position's top-k predictions. This is faster and lighter but approximate. |
| `--prediction_topk` | Number of predicted token ids kept per position in `topk_unique` mode. |
| `--lowest_likelihood_ratio` | Keep only the lowest-likelihood token/response positions for representation building. `0.1` keeps the hardest 10%; `1.0` keeps all positions. |
| `--topk_abs` | LLM selection only: keep the top-k absolute `pb_diff` token dimensions per position before scoring. |
| `--select_ratio` | LLM selection only: fraction of training samples selected from the highest-scoring examples. |
| `--load_path` | Optional PEFT/LoRA adapter loaded on top of `--model_path` for LLM data selection. |
| `--embed_device` / `--device` | Device used for model forward passes. |
| `--score_device` | Device used for pairwise or validation-mean scoring. Defaults to the forward device when omitted in optimized VL/LLM scripts. |
| `--local_files_only` | Load models/tokenizers/processors from the local HuggingFace cache only. |

## Outputs

Default output files include:

```text
result_dict_streaming.json
time_dict_streaming.json
result_dict_vl_optimized.json
time_dict_vl_optimized.json
result_dict_llm_select_optimized.json
```

Text and vision-language valuation write metrics such as:

```text
proposed_auc
proposed_auc_std
proposed_recall
proposed_recall_std
vocab_mode
prediction_topk
```

LLM selection writes selection metadata such as:

```text
num_selected
select_ratio
selected_clean_ratio
validation_vocab_size
val_build_sec
train_score_sec
total_sec
```

## Troubleshooting

- Prefer `topk_unique` when `total_unique` uses too much memory.
- Lower `--prediction_topk`, `--batch_size`, `--batch_size_train`, or `--batch_size_test` if you hit CUDA OOM.
- Lower `--lowest_likelihood_ratio` to score fewer difficult positions.
- Put scoring on a different GPU with `--score_device` when forward-pass memory and scoring memory compete.
- For local/offline runs, add `--local_files_only` and verify that the model, tokenizer, processor, and dataset are already cached or saved on disk.
