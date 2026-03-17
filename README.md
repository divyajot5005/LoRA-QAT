# Quantization-Aware LoRA Prototype

This repository contains a minimal research prototype for quantization-aware PEFT with LoRA.

## Method

For each LoRA-adapted layer, let:

- `M` be the frozen base weight
- `M' = B @ A` be the LoRA update
- `W = M + M'` be the merged deployable weight

The training loss is:

`L = L_task + lambda_q * dist(W, Q(W))`

In this prototype:

- `Q(W)` is a uniform symmetric fake quantizer
- `dist` is mean squared error between `W` and `Q(W)`
- the regularizer is applied only to LoRA-touched weights
- the feature is fully optional and off by default
- there is no second task forward pass, Hessian, or Fisher term

## Project Layout

- `train.py`: synthetic sanity-check trainer
- `train_hf.py`: Hugging Face trainer for real pretrained models
- `quant_lora/config.py`: synthetic config dataclasses and JSON loader
- `quant_lora/hf_config.py`: Hugging Face config dataclasses and JSON loader
- `quant_lora/data.py`: synthetic teacher-student classification task
- `quant_lora/hf_data.py`: Hugging Face dataset loading and tokenization
- `quant_lora/lora.py`: minimal LoRA wrapper used by the synthetic stack
- `quant_lora/hf_model.py`: pretrained model and PEFT LoRA setup
- `quant_lora/quantization.py`: uniform quantizers and shared stats structs
- `quant_lora/peft_regularization.py`: merged-weight quantization regularizer for PEFT layers
- `quant_lora/trainer.py`: synthetic training loop
- `quant_lora/hf_trainer.py`: Hugging Face training loop and evaluation
- `configs/`: synthetic and real-model experiment configs

## Quantization Config Flags

Quantization-aware LoRA regularization is controlled by:

- `enable_quant_lora_regularization`
- `lambda_q`
- `quantizer_type`
- `bit_width`
- `group_size`
- `regularization_frequency`
- `log_per_layer_stats`

Supported `quantizer_type` values:

- `uniform_per_tensor`
- `uniform_per_channel`
- `uniform_groupwise`
- `fp8_e4m3fn`
- `fp8_e5m2`

## Synthetic Sanity Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Baseline sanity run:

```bash
python train.py --config configs/sanity_baseline.json
```

Quantization-aware sanity run:

```bash
python train.py --config configs/sanity_quant_lora.json
```

## Real Model Runs

These runs were tested with the `torch-gpu` conda environment on a 6 GB RTX 4050 Laptop GPU.

DistilBERT on AG News:

```bash
C:\conda-envs\torch-gpu\python.exe train_hf.py --config configs/hf_distilbert_agnews_quant_lora.json
```

GPT-2 on WikiText-2:

```bash
C:\conda-envs\torch-gpu\python.exe train_hf.py --config configs/hf_gpt2_wikitext_quant_lora.json
```

The provided configs are intentionally small and research-friendly:

- DistilBERT: `distilbert-base-uncased`, LoRA on `q_lin` and `v_lin`
- GPT-2: `gpt2`, LoRA on `c_attn` and `c_proj`
- short training runs to verify the merged-weight regularizer on real pretrained models

## H100 Experiment Suite

For larger-scale testing on a single H100, use the one-shot suite runner:

```bash
python run_h100_quant_lora_suite.py --output-root outputs/h100_suite
```

The suite is set up for:

- 2 approximately 1B models:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- 2 7B to 8B models:
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `Qwen/Qwen2.5-7B-Instruct`
- 2 14B models:
  - `Qwen/Qwen2.5-14B-Instruct`
  - `Qwen/Qwen2.5-Coder-14B-Instruct`

And 3 real-world task families:

- `customer_support`: supervised response generation on Bitext customer support
- `instruction_tuning`: supervised instruction tuning on Dolly 15k
- `domain_adaptation_code`: code-domain adaptation on CodeSearchNet Python

For each model-task pair, the script prepares and optionally runs:

- `before_ft`: evaluation before any finetuning
- `no_quant_ft`: LoRA finetuning without quantization regularization
- `quant_int4_ft`: merged-weight regularized LoRA targeting INT4 deployability
- `quant_fp8_ft`: merged-weight regularized LoRA targeting FP8 deployability

The H100 configs are optimized around:

- `torch_dtype="bfloat16"`
- `mixed_precision="bf16"`
- `attn_implementation="sdpa"`
- gradient checkpointing enabled
- larger step counts than the local smoke tests

Useful options:

```bash
python run_h100_quant_lora_suite.py --write-only
python run_h100_quant_lora_suite.py --models qwen25_14b qwen25_coder_14b --tasks instruction_tuning
python run_h100_quant_lora_suite.py --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft
python run_h100_quant_lora_suite.py --export-fp8-offline
```

Outputs:

- run configs: `outputs/h100_suite/configs/`
- run directories: `outputs/h100_suite/runs/`
- summary CSV: `outputs/h100_suite/summary.csv`
- summary Markdown: `outputs/h100_suite/summary.md`
- generated serving commands: `outputs/h100_suite/serving_recipes.md`

Optional extra packages for large-model training and serving:

```bash
pip install bitsandbytes vllm
pip install auto-fp8  # optional, only for offline FP8 export
```

Note: `vllm`, `bitsandbytes`, and `auto-fp8` are most practical on Linux. The training suite itself remains standard `transformers` + `peft`.

## Quantized Serving

After a run finishes, the adapter is saved under each run directory at `adapter/`.

You can launch a quantized serving stack directly with:

```bash
python serve_quantized_adapter.py ^
  --base-model Qwen/Qwen2.5-7B-Instruct ^
  --adapter-dir outputs/h100_suite/runs/qwen25_7b__instruction_tuning__quant_int4_ft/adapter ^
  --quantization int4 ^
  --served-model-name qwen25-7b-int4-lora
```

FP8 example:

```bash
python serve_quantized_adapter.py ^
  --base-model Qwen/Qwen2.5-14B-Instruct ^
  --adapter-dir outputs/h100_suite/runs/qwen25_14b__customer_support__quant_fp8_ft/adapter ^
  --quantization fp8 ^
  --served-model-name qwen25-14b-fp8-lora
```

The suite also writes ready-to-run `vllm` commands into `outputs/h100_suite/serving_recipes.md`, plus shell snippets for INT4 and FP8.

## Expected Logging

Both trainers log:

- `task_loss`
- `quant_reg_loss`
- `avg_distance_to_grid`
- validation metrics
- optional per-layer quantization stats

The Hugging Face trainer also logs approximate peak GPU memory.

## Runs Completed On This Machine

Using `C:\conda-envs\torch-gpu\python.exe` on the local GPU:

- DistilBERT / AG News: `final eval_loss=1.156527`, `final eval_accuracy=0.460526`
- GPT-2 / WikiText-2: `final eval_loss=4.397587`, `final eval_perplexity=81.254546`

## Limitations

- This is a research prototype, not a production training stack.
- The fake quantizer is simple and symmetric; there is no calibration, clipping search, or learned quantizer.
- The FP8 regularizer uses a fake projection through PyTorch FP8 dtypes; deploy-time FP8 kernels and scaling behavior may differ from training-time regularization.
- The regularizer operates on the currently merged LoRA-touched weights only.
- The Hugging Face path currently assumes PEFT LoRA layers with 2D base weights.
- GPT-2 uses PEFT's `Conv1D` LoRA support, so groupwise dimensions follow the merged parameter layout used by PEFT.
- The provided real-model configs are small smoke tests, not tuned baselines.
- The H100 suite is configured for open-weight models and public datasets, but some models may require a Hugging Face login or updated license acceptance before download.
- The large-model suite is designed for a single high-memory GPU, but exact feasible batch sizes still depend on sequence length, driver stack, and kernel availability.
- Windows Hugging Face caching may warn about symlink support; this affects cache efficiency, not correctness.
"# LoRA-QAT" 
