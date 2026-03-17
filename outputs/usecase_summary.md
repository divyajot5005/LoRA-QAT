# Real-World Use Case Summary

## DistilBERT: Sentiment Classification

- Task: IMDb review sentiment classification
- Model: `distilbert-base-uncased`
- Config: `configs/hf_distilbert_imdb_quant_usecase.json`
- Final eval loss: `0.291824`
- Final eval accuracy: `0.874600`
- Peak GPU memory (MB): `1152.87`
- Metrics file: `outputs/distilbert_imdb_quant_usecase/final_metrics.json`

## GPT-2: Customer Support Response Generation

- Task: customer support response generation
- Model: `gpt2`
- Dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Config: `configs/hf_gpt2_customer_support_quant_usecase.json`
- Final eval loss: `2.070621`
- Final eval perplexity: `7.929745`
- Peak GPU memory (MB): `2803.62`
- Metrics file: `outputs/gpt2_customer_support_quant_usecase/final_metrics.json`
- Sample generations: `outputs/gpt2_customer_support_quant_usecase/sample_generations.json`

## Notes

- Both runs use merged-weight quantization regularization with LoRA and keep the regularizer defined in deployable weight space.
- The GPT-2 customer support run produced substantially more coherent qualitative outputs than the broader Dolly instruction-tuning run on this same hardware.
