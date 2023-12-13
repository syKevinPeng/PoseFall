#!/bin/bash
python train.py \
--wandb_exp_name exp0 \
--wandb_exp_description "first training- no data aug, only on data recorded in the first session" \
--wandb_mode online