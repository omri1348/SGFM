#!/bin/sh

MODEL_PATH=XXX
NUM_STEPS=XXX
SLOPE_K=XXX
SLOPE_X=XXX
echo "Evaluating model at $MODEL_PATH with $NUM_STEPS sampling steps, slope_k=$SLOPE_K, slope_x=$SLOPE_X"
python -u src/sgfm/model_eval.py $MODEL_PATH --num_steps $NUM_STEPS --slope_k $SLOPE_K --slope_x $SLOPE_X
