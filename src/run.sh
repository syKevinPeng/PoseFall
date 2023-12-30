#!/bin/bash
# for training
exp_name="exp4"

# python train.py \
# --wandb_exp_name "$exp_name" \
# --wandb_exp_description "Formal Training: With spline loss and smoothness loss. With data aug. Pretrained weights on Encoder. KL divergence loss: 0.1. Decreased smoothness loss    " \
# --wandb_mode online \
# --epochs 1000 \
# --batch_size 16 \

# for evaluation
python evaluate.py \
--wandb_exp_name "$exp_name Eval" \
--wandb_mode disabled \
--wandb_exp_description "Evaluation on $exp_name" \
--num_to_gen 1 \
--output_path "./gen_results" \
--ckpt_path "/home/siyuan/research/PoseFall/src/wandb/run-20231230_000808-qhs5vrwk/files/2023-12-30_00-08-08_exp4/epoch_999.pth"

