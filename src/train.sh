#!/bin/bash
python train.py \
--wandb_exp_name test \
--wandb_exp_description "first training- no data aug, only on data recorded in the first session" \
--wandb_mode disabled \
--epochs 2000 \
--batch_size 16
