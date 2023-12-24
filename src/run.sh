#!/bin/bash
# for training
exp_name="exp0"

# python train.py \
# --wandb_exp_name "$exp_name" \
# --wandb_exp_description "Formal Training: With spline loss and smoothness loss. No data aug" \
# --wandb_mode online \
# --epochs 2000 \
# --batch_size 16 \

# for evaluation
python evaluate.py \
--wandb_exp_name "$exp_name" \
--wandb_mode online \
--wandb_exp_description "Evaluation on $exp_name" \
--ckpt_path "/home/siyuan/research/PoseFall/src/wandb/run-20231221_215107-4o2q6lod/files/2023-12-21_21-51-07_exp0/epoch_1999.p5"

