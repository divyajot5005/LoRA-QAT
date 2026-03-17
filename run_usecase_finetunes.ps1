$python = 'C:\conda-envs\torch-gpu\python.exe'
& $python train_hf.py --config configs\hf_distilbert_imdb_quant_usecase.json
& $python train_hf.py --config configs\hf_gpt2_customer_support_quant_usecase.json
