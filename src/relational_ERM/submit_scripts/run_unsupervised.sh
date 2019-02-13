#!/bin/bash

DATA_DIR=../../data/pokec/regional_subset/
OUTPUT_DIR=../../output/unsupervised_pokec_regional_embeddings/

python -m relational_erm.rerm_model.run_classifier \
      --do_train \
      --num_train_steps 15000 \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --batch-size 256 \
      --embedding_learning_rate=5e-3 \
      --save_checkpoints_steps=1000 \
      --unsupervised