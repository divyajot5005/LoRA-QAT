# Quantization-through-LoRA

This repository studies a simple question:

How do we finetune a model so that the **quantized model we actually deploy** is better, not just the floating-point model we train?

The project started with LoRA and later added a 1B full-finetuning extension. The core idea is to regularize deployable weights toward a target quantizer during training, then evaluate the model **after merge and proxy quantization**.

## What The Project Actually Shows

The main completed result is not “quantization-aware training always helps.”

The stronger and more defensible result is:

- for `INT8` and mild `INT4`, gains over ordinary finetuning followed by post-training quantization are usually tiny
- for harsher targets such as `2-bit` and ternary, the same idea can produce meaningful post-quantization improvements

Representative completed wins:

- `Llama 3.2 1B` LoRA, `2-bit`:
  - baseline merged-then-quantized loss: `14.27055`
  - best regularized run: `12.14152`
  - improvement: `-2.12903`

- `Llama 3.2 1B` LoRA, ternary:
  - baseline merged-then-quantized loss: `10.68995`
  - best regularized run: `10.26500`
  - improvement: `-0.42495`

- `Qwen 0.6B` full-FT, `INT4`:
  - baseline merged-then-quantized loss: `3.68961`
  - best regularized run: `3.62244`
  - improvement: `-0.06717`

- `Qwen 0.6B` full-FT, `2-bit`:
  - baseline merged-then-quantized loss: `18.42231`
  - best regularized run: `17.74188`
  - improvement: `-0.68044`

The completed `code_search_net` LoRA slice is useful mainly as a stability check:

- regularization usually preserves task quality
- improvements exist for some model/target pairs
- but most gains there are small

## Core Method

For a deployable weight matrix `W`, training uses:

`L = L_task + lambda_q * R(W, Q(W))`

where:

- `Q(W)` is the target quantizer
- `R` is usually a weight-space MSE surrogate
- evaluation is done on the merged-and-quantized model, not just the float model

In the LoRA setting:

- `W = W0 + BA`
- regularization is applied in merged deployable weight space

In the full-FT setting:

- the same idea is applied directly to trainable full-model linear weights

## Experiment Types In This Repo

### 1. LoRA experiments

Main runner:

- `C:\Work\Papers\Quantization-through-LoRA\run_h100_quant_lora_suite.py`

Completed LoRA targets include:

- `no_quant_ft`
- `quant_int4_ft`
- `quant_fp8_ft`
- `quant_int4_act_ft`
- `quant_fp8_act_ft`
- newer low-bit phases:
  - `quant_int2_ft`
  - `quant_ternary_ft`

Main LoRA tasks used in this repo:

- `instruction_tuning`
- `domain_adaptation_code`

### 2. Full-finetuning experiments

Main runner:

- `C:\Work\Papers\Quantization-through-LoRA\run_full_ft_quant_sweep.py`

Completed full-FT sweeps use:

- `INT8`
- `INT4`
- `2-bit`
- ternary

with multiple `lambda_q` values per target.

### 3. Proxy quantized evaluation

The key evaluation is:

- take the trained model or merged LoRA adapter
- quantize it with a proxy quantizer
- evaluate downstream task loss / perplexity

Relevant script:

- `C:\Work\Papers\Quantization-through-LoRA\evaluate_quantized_proxy.py`

This is the metric that matters for the paper.

## Where The Results Are

### Main local proxy-eval summaries

- `C:\Work\Papers\Quantization-through-LoRA\outputs\a6000_proxy_eval\all_proxy_eval_summary.csv`
- `C:\Work\Papers\Quantization-through-LoRA\outputs\a6000_proxy_eval\lora_proxy_backfill_summary.csv`
- `C:\Work\Papers\Quantization-through-LoRA\outputs\a6000_proxy_eval\fullft_proxy_backfill_summary.csv`

Important columns:

- `float_loss`: float-model task loss
- `quant_loss`: post-quantization task loss
- `eval_loss_delta`: `quant_loss - float_loss`
- `active_primary_metric`: float-model task metric
- `proxy_primary_metric`: post-quantization task metric
- `primary_metric_delta`: degradation after quantization

### Earlier low-bit LoRA result set

The strongest ultra-low-bit LoRA results used in the paper came from earlier completed sweeps under:

- `C:\Work\Papers\Quantization-through-LoRA\outputs`

### Paper draft

- `C:\Work\Papers\Quantization-through-LoRA\paper_acl_workshop\main.tex`

## Quick Interpretation Guide

When reading a CSV row:

- lower `quant_loss` is better
- lower `eval_loss_delta` is better
- lower `primary_metric_delta` is better

The main baseline for a target is usually:

- `no_quant_ft` for LoRA
- `no_quant` for full FT

The right question is:

“Does some regularized run achieve lower post-quantization task loss than ordinary finetuning followed by quantization?”

## Project Structure

Main training and evaluation files:

- `C:\Work\Papers\Quantization-through-LoRA\train_hf.py`
- `C:\Work\Papers\Quantization-through-LoRA\run_h100_quant_lora_suite.py`
- `C:\Work\Papers\Quantization-through-LoRA\run_full_ft_quant_sweep.py`
- `C:\Work\Papers\Quantization-through-LoRA\evaluate_quantized_proxy.py`

Core library code:

- `C:\Work\Papers\Quantization-through-LoRA\quant_lora\hf_model.py`
- `C:\Work\Papers\Quantization-through-LoRA\quant_lora\hf_trainer.py`
- `C:\Work\Papers\Quantization-through-LoRA\quant_lora\peft_regularization.py`
- `C:\Work\Papers\Quantization-through-LoRA\quant_lora\quantization.py`
- `C:\Work\Papers\Quantization-through-LoRA\quant_lora\hf_config.py`

Paper and manuscript:

- `C:\Work\Papers\Quantization-through-LoRA\paper_acl_workshop\main.tex`

## Running Experiments

### LoRA suite

```bash
python C:\Work\Papers\Quantization-through-LoRA\run_h100_quant_lora_suite.py --output-root C:\Work\Papers\Quantization-through-LoRA\outputs\h100_suite
```

### Full-FT sweep

```bash
python C:\Work\Papers\Quantization-through-LoRA\run_full_ft_quant_sweep.py --output-root C:\Work\Papers\Quantization-through-LoRA\outputs\fullft
```

### Proxy evaluation

```bash
python C:\Work\Papers\Quantization-through-LoRA\evaluate_quantized_proxy.py --help
```

## Main Takeaways

- The project is strongest as a **deployment-aware customization** study, not as a universal LoRA improvement claim.
- The method is most useful when quantization damage is large enough to matter.
- Ultra-low-bit results are the strongest evidence.
- Mild-precision improvements are often too small to matter.
- Supporting A6000 runs are useful for scope and stability, but the headline result is still the low-bit deployment story.

## Current Limitations

- Weight-space MSE is only a proxy for deployed model quality.
- Some branches are incomplete or contain failed runs; the paper should use only completed post-quantization evaluations.
- Gains are inconsistent across models and precisions.
- `INT8` is often too easy for this method to show meaningful value.
- Very low-bit runs can improve deployed quality while still hurting float-side quality, so target-dependent tradeoffs matter.
