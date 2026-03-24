# Local LoRA-QAT Benchmark Summary

## distilbert_agnews

- `baseline` targeting `int4`: active eval_accuracy=0.871053, proxy_quantized eval_accuracy=0.872368, delta=0.001316, raw_quant_error=2.605746e-05
- `baseline` targeting `fp8`: active eval_accuracy=0.871053, proxy_quantized eval_accuracy=0.867105, delta=-0.003947, raw_quant_error=2.605746e-05
- `qat_int4` targeting `int4`: active eval_accuracy=0.871053, proxy_quantized eval_accuracy=0.872368, delta=0.001316, raw_quant_error=2.605746e-05
- `qat_fp8` targeting `fp8`: active eval_accuracy=0.871053, proxy_quantized eval_accuracy=0.867105, delta=-0.003947, raw_quant_error=1.531128e-06
## gpt2_customer_support

- `baseline` targeting `int4`: active eval_perplexity=11.003031, proxy_quantized eval_perplexity=13.785045, delta=2.782015, raw_quant_error=2.355159e-04
- `baseline` targeting `fp8`: active eval_perplexity=11.003031, proxy_quantized eval_perplexity=10.951927, delta=-0.051104, raw_quant_error=2.355159e-04
- `qat_int4` targeting `int4`: active eval_perplexity=11.003130, proxy_quantized eval_perplexity=13.790084, delta=2.786955, raw_quant_error=2.355159e-04
- `qat_fp8` targeting `fp8`: active eval_perplexity=11.003482, proxy_quantized eval_perplexity=10.953228, delta=-0.050255, raw_quant_error=1.202855e-05
