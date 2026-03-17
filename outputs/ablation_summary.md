# Long Ablation Summary

## DistilBERT on AG News

- Baseline config: `configs/hf_distilbert_agnews_baseline_long.json`
- Quant config: `configs/hf_distilbert_agnews_quant_long.json`
- Baseline final eval loss: `0.323899`
- Baseline final eval accuracy: `0.900658`
- Quant final eval loss: `0.323810`
- Quant final eval accuracy: `0.900658`
- Baseline peak GPU memory (MB): `735.23`
- Quant peak GPU memory (MB): `986.49`

## GPT-2 on WikiText-2

- Baseline config: `configs/hf_gpt2_wikitext_baseline_long.json`
- Quant config: `configs/hf_gpt2_wikitext_quant_long.json`
- Baseline final eval loss: `3.817888`
- Baseline final eval perplexity: `45.508006`
- Quant final eval loss: `3.817888`
- Quant final eval perplexity: `45.508009`
- Baseline peak GPU memory (MB): `1168.10`
- Quant peak GPU memory (MB): `2733.31`

## Interpretation

- The longer end-to-end runs are stable on this GPU and the quant-regularized runs preserve task quality relative to baseline.
- In these settings, the quantization regularizer is active but gentle; it does not measurably degrade final metrics.
- These are credible end-to-end ablation runs, but they are still small-scope laptop experiments rather than publication-scale sweeps.
