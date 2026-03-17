# Serving Recipes

INT4 serving uses vLLM BitsAndBytes in-flight quantization.
FP8 serving uses vLLM online dynamic FP8 quantization by default; for lower load-time memory, optionally export offline FP8 with AutoFP8.
