#!/bin/bash

export PYTHONPATH=/home/urp_jwl/.vscode-server/data/kkp_drug

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 ./scripts/train.py \
    --model_name 'DeepChem/ChemBERTa-77M-MLM' \
    --save_dir './chemberta' \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 3e-3 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-3 \
    --logging_steps 20 \
    --bert_out_dim 1 \
    --k_fold 5 \
    --mol_f_dim 2083 \