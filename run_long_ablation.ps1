$python = 'C:\conda-envs\torch-gpu\python.exe'
& $python train_hf.py --config configs\hf_distilbert_agnews_baseline_long.json
& $python train_hf.py --config configs\hf_distilbert_agnews_quant_long.json
& $python train_hf.py --config configs\hf_gpt2_wikitext_baseline_long.json
& $python train_hf.py --config configs\hf_gpt2_wikitext_quant_long.json
