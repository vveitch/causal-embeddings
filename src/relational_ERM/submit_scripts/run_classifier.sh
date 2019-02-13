#!/bin/bash

INIT_DIR=../../output/unsupervised_pokec_regional_embeddings
INIT_FILE=$INIT_DIR/model.ckpt-15000
DATA_DIR=../../data/pokec/regional_subset
OUTPUT_DIR=../../output/pokec_prediction/settingA/seed0/

python -m relational_erm.rerm_model.run_classifier \
  --seed 0 \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --label_task_weight 5e-3\
  --num_train_steps 1000 \
  --batch-size 100 \
  --embedding_learning_rate 5e-3 \
  --global_learning_rate 5e-3 \
  --save_checkpoints_steps=500 \
  --num_train_steps 1000 \
  --unsupervised \
  --label_pred \
  --proportion-censored 0.1 \
  --simulation_setting A \
  --discretize_simulation 1 \
  --easy_treatment_simulation 1
#  --init_checkpoint=$INIT_FILE