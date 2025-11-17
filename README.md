## SGFM

### Setup
Run this script to initialize the virtual environment (uv is required).
```
bash setup_uv.sh
```
To activate the virtual environemnt run `source .venv/bin/activate`.
### Data Setup
Run the following scripts to preprocess the crystal datasets. This step is required before training and evaluation.
```
bash scripts/data_setup.sh
bash scripts/crystal_pkl.sh
```
### Training

The following scripts can be used to reproduce the SGFM models from the paper (CSP and DNG) trained on the MP-20 dataset.
```
bash scripts/run_csp.sh
bash scripts/run_dng.sh
```

A custom training session can be initiated with the following command.
```
python src/sgfm/run.py expaname=<XX> data=<XX> ...
```
Training parameters such as dataset, model, and optimization settings can be modified via the Hydra API.
The `data` tag can be selected from perov, mp_20, mpts_52 and carbon.

### Evaluation
CSP/DNG evaluation is performed by running the `scripts/eval.sh` script. To execute it, the user must specify a checkpoint file (.ckpt) to evaluate, the number of generation steps, and the anti-annealing parameters (slope_k and slope_x). Setting both parameters to 0 disables anti-annealing.
```
MODEL_PATH=XXX
NUM_STEPS=XXX
SLOPE_K=XXX
SLOPE_X=XXX
echo "Evaluating model at $MODEL_PATH with $NUM_STEPS sampling steps, slope_k=$SLOPE_K, slope_x=$SLOPE_X"
python -u src/sgfm/model_eval.py $MODEL_PATH --num_steps $NUM_STEPS --slope_k $SLOPE_K --slope_x $SLOPE_X
```
The type of evaluation (CSP/DNG) is determined by the model config.

