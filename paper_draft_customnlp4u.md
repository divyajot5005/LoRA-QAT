# Quantization-Aware Customization via Deployable-Weight Regularization

## Draft for CustomNLP4U Workshop

### Title Options

1. Quantization-Aware Customization via Deployable-Weight Regularization
2. Training for Quantized Deployment: A Simple Regularization Strategy for PEFT and Finetuning
3. Beyond Post-Training Quantization: Quantization-Aware Customization with Merged-Weight Regularization

### Abstract

Parameter-efficient finetuning and full finetuning are often evaluated in full precision, while deployment frequently happens in lower precision such as INT8, INT4, or even more aggressive low-bit formats. This creates a mismatch between the optimization target during customization and the model actually used at inference time. We study a simple quantization-aware customization objective that regularizes deployable weights toward a target quantizer during training. Concretely, we add a quantization regularization term on the deployable weight space, rather than applying quantization only as a post-processing step after finetuning. We evaluate this idea primarily in LoRA-based customization and use post-merge proxy quantization to measure downstream task quality under multiple quantization targets. Across standard INT8 and INT4 settings, the gains over LoRA followed by post-training quantization are small. However, at more aggressive ultra-low-bit settings, including 2-bit and ternary quantization, the same regularization produces materially better post-quantization task loss than a no-regularizer baseline. These results suggest that simple quantization-aware regularization has limited value when the quantization gap is already small, but becomes more useful when deployment constraints are harsher. We also describe an extension of the same idea beyond LoRA to full finetuning, which broadens the applicability of deployment-aware customization.

### 1. Introduction

Model customization is typically optimized for full-precision quality, while production deployment is often constrained by memory and latency requirements that require quantization. In practice, this means that a model is customized first and quantized later, even though the final deployed model is the quantized one. This train-deploy mismatch is especially relevant for efficient customization methods such as LoRA, where practitioners often merge adapters into base weights and then quantize for serving.

This mismatch motivates a simple question: can we customize a model in a way that directly improves the quality of the quantized model we will eventually deploy?

In this work, we study a simple quantization-aware customization method based on deployable-weight regularization. During customization, we add a penalty that encourages trained weights to remain close to a target quantization grid. For LoRA, the regularizer is applied to the merged deployable weights, i.e., the base weight after adding the low-rank update. This keeps the optimization target closer to the final artifact used in deployment.

Our experiments show a nuanced picture. For moderate quantization settings such as INT8 and INT4, the method is usually neutral or only marginally helpful compared with a strong baseline that performs standard LoRA finetuning followed by post-training quantization. In contrast, under harsher 2-bit and ternary targets, the same regularizer yields meaningful improvements in post-quantization task loss. The main takeaway is not that simple weight-space regularization uniformly solves quantization-aware customization, but that its value depends strongly on how much quality is lost under the target deployment precision.

This is a useful result for customization settings targeted by the CustomNLP4U workshop. Customization methods should ideally be evaluated not only on task performance in floating point, but also on the quality of the actual deployed model under real hardware-aware constraints.

### 2. Contributions

We make the following contributions:

1. We propose a simple quantization-aware customization objective based on regularizing deployable weights toward a target quantizer during finetuning.
2. We evaluate the method using LoRA and measure the downstream task loss of the merged-and-quantized model, rather than only reporting floating-point performance or raw quantization error.
3. We show that the method offers little advantage when the quantization gap is already small (INT8, modest INT4), but becomes more beneficial in harsher ultra-low-bit settings such as 2-bit and ternary quantization.
4. We extend the same conceptual framework beyond LoRA to full finetuning, enabling broader deployment-aware customization, and present this as an ongoing extension of the current study.

### 3. Method

#### 3.1 Standard LoRA customization

Let a pretrained weight matrix be denoted by \(W_0\). LoRA parameterizes the finetuned weight as

\[
W = W_0 + \Delta W, \quad \Delta W = BA,
\]

where \(A\) and \(B\) are low-rank trainable matrices. In deployment, the adapter is merged into the base weight and the merged weight is then quantized.

#### 3.2 Deployable-weight regularization

We add a regularization term that penalizes the distance between the deployable merged weight \(W\) and its quantized counterpart \(Q(W)\):

\[
\mathcal{L} = \mathcal{L}_{task} + \lambda_q \cdot \mathcal{R}(W, Q(W)).
\]

In the current implementation, the primary regularizer is a weight-space mean squared error objective:

\[
\mathcal{R}(W, Q(W)) = \|W - Q(W)\|_2^2.
\]

For LoRA, this regularizer is applied to LoRA-touched modules in merged weight space. This keeps the method simple and compatible with existing PEFT stacks. We also explored activation-aware variants and broader full-finetuning variants of the same idea, but the strongest completed evidence currently comes from the merged-weight formulation.

#### 3.3 Evaluation protocol

The key metric in this work is not the regularization loss itself, nor only the floating-point task loss. Instead, for every trained run we evaluate:

1. floating-point task loss after finetuning,
2. task loss after merging adapters and applying proxy quantization,
3. the quantization gap, defined as:

\[
\Delta_{quant} = \mathcal{L}_{quantized} - \mathcal{L}_{float}.
\]

This is the relevant deployment metric because it measures the quality of the actual quantized model used after customization.

### 4. Related Work

Our work is related to several lines of prior work:

- Parameter-efficient finetuning via LoRA and related adapter methods.
- Quantization-aware training and low-bit deployment techniques.
- Prior work on quantization-aware PEFT, including methods that modify LoRA parameterization or jointly design quantization and adaptation.

The closest distinction is that our method does not redesign the LoRA operator itself. Instead, it keeps the standard customization recipe largely intact and adds a lightweight regularization term in deployable weight space. This makes the method closer to a plug-in deployment-aware training objective than to a new quantization-specific adapter architecture.

At the same time, our experiments suggest that this simplicity has limits: a naive weight-space regularizer is not enough to produce strong gains when the target quantizer already induces only mild degradation.

### 5. Experimental Setup

#### 5.1 Tasks

We primarily evaluate causal language modeling customization on instruction-following data. Our current completed experiments use the `databricks/databricks-dolly-15k` dataset for instruction tuning. We also queue a second-task robustness evaluation on code-domain adaptation using `code_search_net` to verify whether the same trends hold outside the instruction-tuning setting.

#### 5.2 Models

The main completed LoRA experiments currently cover:

- Llama 3.2 1B Instruct
- Gemma 3 1B IT
- TinyLlama 1.1B Chat
- additional small-model exploratory runs

The active robustness and scale-extension queue currently targets:

- Llama 3.2 1B
- Llama 3.2 3B
- Llama 3.1 8B
- Qwen3 0.6B / 4B / 8B
- Gemma 3 1B / 4B and Gemma 2 9B

For the workshop submission, the final version should keep only the models with completed and stable results.

#### 5.3 Quantization targets

We evaluate the following proxy deployment targets:

- INT8
- INT4
- 2-bit
- ternary (approximately 1.58-bit effective representation)

The most informative results come from the harsher low-bit settings, where the quantization gap is large enough for the regularizer to matter.

#### 5.4 Baselines

We compare against:

1. standard LoRA finetuning without quantization regularization,
2. post-training proxy quantization of the merged LoRA model,
3. quantization-regularized LoRA finetuning under matched target quantizers.

The principal comparison is:

> task loss after merging and quantizing the regularized model
>
> versus
>
> task loss after merging and quantizing the no-regularizer LoRA baseline.

### 6. Results

#### 6.1 INT8 and INT4 provide only marginal gains

For moderate quantization settings, the method gives at best very small improvements over the post-training quantization baseline.

For example, on Llama 3.2 1B:

- best INT4 post-quantized loss with regularization: `1.98630`
- no-regularizer LoRA baseline after INT4 quantization: `1.98747`

and:

- best INT8 post-quantized loss with regularization: `1.85132`
- no-regularizer LoRA baseline after INT8 quantization: `1.85160`

These gains are directionally positive but too small to be the main story of the paper. This indicates that when the deployment quantizer is already relatively benign, a simple weight-space regularizer has limited headroom.

#### 6.2 Ultra-low-bit settings show meaningful improvements

The picture changes when the deployment target becomes harsher.

For Llama 3.2 1B under 2-bit proxy quantization:

- no-regularizer LoRA baseline after quantization: `14.27055`
- regularized run with \(\lambda_q = 1 \times 10^5\): `14.04537`
- regularized run with \(\lambda_q = 3 \times 10^5\): `13.41306`
- regularized run with \(\lambda_q = 1 \times 10^6\): `12.14152`

For ternary proxy quantization:

- no-regularizer LoRA baseline after quantization: `10.68995`
- regularized run with \(\lambda_q = 1 \times 10^4\): `10.61170`
- regularized run with \(\lambda_q = 3 \times 10^4\): `10.59449`
- regularized run with \(\lambda_q = 1 \times 10^5\): `10.43695`
- regularized run with \(\lambda_q = 3 \times 10^5\): `10.26500`

These improvements are materially larger than the INT4 and INT8 gains. This supports the view that deployment-aware regularization is most useful when the post-training quantization baseline has a substantial quality gap.

#### 6.3 Trade-off between floating-point quality and quantized quality

The strongest low-bit improvements do not always come for free. In some 2-bit settings, higher regularization strength improves post-quantized loss while degrading floating-point performance. This suggests that the relevant optimization objective depends on the deployment target:

- if deployment is at INT8 or mild INT4, preserving full-precision task quality may be sufficient,
- if deployment is at 2-bit or ternary precision, a small loss in floating-point quality may be acceptable if the quantized deployed model improves substantially.

This is a deployment trade-off, not merely a training trade-off, and it should be reported explicitly.

### 7. Analysis

Our experiments also reveal a limitation of simple weight-space regularization. In many settings, the raw quantization error or quantization regularization loss decreases during training, but the downstream task loss of the merged-and-quantized model does not improve by much. This means that reducing \(\|W - Q(W)\|^2\) is not always well aligned with reducing post-quantization task loss.

This mismatch is most visible in INT8 and mild INT4 regimes, where:

- raw quantization error decreases,
- the weighted regularizer can dominate training if \(\lambda_q\) is large,
- but the actual post-quantization task metric improves only slightly or not at all.

This leads to an important lesson: optimization in weight space alone may be too indirect for settings where the target quantizer already preserves model quality well. More output-aligned objectives, such as logit- or activation-level consistency between float and quantized models, are a promising next step.

### 8. Broader Customization Perspective

Although the strongest completed evidence in this paper comes from LoRA, the underlying idea is not specific to LoRA. The same principle applies to full finetuning:

- customize the model for the target task,
- while regularizing the deployable weights toward a target quantizer,
- and evaluate the actual post-quantization task loss of the finetuned model.

We implemented this broader full-finetuning version and use it as an extension path beyond the current LoRA-centric results. This is important for the customization community because the proposed objective is better understood as a deployment-aware customization framework than as a LoRA-specific method.

### 9. Limitations

This work has several limitations.

First, the current completed results are single-seed. We do not present multi-seed estimates in the current draft. The intended workshop claim is therefore based on clear effect size in low-bit settings rather than variance-sensitive small gains.

Second, the strongest evidence currently comes from proxy quantization rather than production kernels on every hardware stack. This is appropriate for a customization study, but future work should validate these results with deployment-time runtimes and serving implementations.

Third, the simple weight-space regularizer is not uniformly effective. In particular, it appears insufficient to produce strong advantages in easy quantization regimes. This is a useful negative result, but it also motivates richer deployment-aware training objectives.

Fourth, some scale-extension experiments are still running at the time of this draft. The final workshop version should include only completed results and should remove any placeholder model families that do not finish in time.

### 10. Conclusion

We studied a simple deployment-aware customization strategy that regularizes finetuned weights toward a target quantizer during training. The main result is conditional but clear: simple deployable-weight regularization offers little advantage when the target quantizer already induces only modest degradation, but becomes meaningfully more useful in harsher ultra-low-bit settings such as 2-bit and ternary quantization.

This result matters for practical customization workflows. If the deployed model will be aggressively quantized, then evaluating only full-precision finetuned quality is not enough. Customization should be assessed in the precision regime in which the model will actually be used.

Our findings also suggest a broader direction for future work: moving from weight-space regularization toward output-aligned deployment-aware customization objectives that more directly optimize the quality of the final quantized model.

### References Placeholder

The final version should include references to:

- LoRA
- QLoRA
- QA-LoRA
- LoftQ
- quantization-aware training for language models
- low-bit and ternary quantization papers relevant to deployment-aware LLM finetuning

### Tables Placeholder

#### Table 1. Main instruction-tuning results

Include:

- model
- quantizer target
- float loss
- quantized loss
- quantization gap
- baseline versus regularized

#### Table 2. Ultra-low-bit improvement table

Include:

- model
- target precision
- \(\lambda_q\)
- float loss
- post-quantization loss
- delta versus no-regularizer baseline

#### Table 3. Robustness on second task

Include:

- Llama 1B / 3B / 8B
- task: domain adaptation code
- same metrics as Table 1

### Appendix Placeholder

Include:

- exact hyperparameters
- model-specific LoRA target modules
- dataset splits
- proxy quantization implementation details
- compute budget and hardware
- additional ablations on \(\lambda_q\)
