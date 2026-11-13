#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

DEFAULT_DEVICE="0"
CUDA_DEVICE="${1:-$DEFAULT_DEVICE}"

echo "use device: cuda:${CUDA_DEVICE}"

python train_gan.py --mode re_go_ba --data_set bitcoinalpha --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 256 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 7 --n_epochs 20 --save_auc 0.48 --train_ratio 0.7
python train_gan.py --mode re_go_wiki --data_set wikipedia --lambda_3 0.3 --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 256 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 7 --n_epochs 20 --save_auc 0.78 --save_auc_rf --train_ratio 0.7 
python train_gan.py --mode re_go_redi --data_set reddit --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 64 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 5 --n_epochs 20 --num_tokens 1024 --hqb_len 5 --hsb_len 3 --n_neighbors 64 --residual_w 0.1  --save_auc 0.67 --train_ratio 0.7
python train_gan.py --mode re_go_bo --lambda_temporal 0.5 --data_set bitcoinotc --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 256 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 7 --n_epochs 20 --save_auc 0.7 --train_ratio 0.7
python train_gan.py --mode re_go_mooc --lambda_3 0.3 --data_set mooc --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 256 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 7 --n_epochs 20 --save_auc 0.79 --train_ratio 0.7 
python train_gan.py --mode re_go_ama --lambda_temporal 0.1 --data_set amazon --llm_model Qwen3-0_6B --llm_dim 1024 --batch_size 256 --hidden_dim 256 --device cuda:${CUDA_DEVICE} --learning_rate_drop_step 7 --n_epochs 20 --save_auc 0.6 --train_ratio 0.7
