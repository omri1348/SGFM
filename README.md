## SGFM

Official Implementation of the paper "Space Group Conditional Flow Matching".

### Abstract
Inorganic crystals are periodic, highly-symmetric arrangements of atoms in three-dimensional space. Their structures are constrained by the symmetry operations of a crystallographic space group and restricted to lie in specific affine subspaces known as Wyckoff positions. The frequency an atom appears in the crystal and its rough positioning are determined by its Wyckoff position. Most generative models that predict atomic coordinates overlook these symmetry constraints, leading to unrealistically high populations of proposed crystals exhibiting limited symmetry.
We introduce Space Group Conditional Flow Matching, a novel generative framework that samples significantly closer to the target population of highly-symmetric, stable crystals. We achieve this by conditioning the entire generation process on a given space group and set of Wyckoff positions; specifically, we define a conditionally symmetric noise base distribution and a group-conditioned, equivariant, parametric vector field that restricts the motion of atoms to their initial Wyckoff position. Our form of group-conditioned equivariance is achieved using an efficient reformulation of group averaging tailored for symmetric crystals. Importantly, it reduces the computational overhead of symmetrization to a negligible level.
We achieve state of the art results on crystal structure prediction and de novo generation benchmarks. We also perform relevant ablations.

Arxiv paper can be found [here](https://www.arxiv.org/abs/2509.23822) 

### Setup
Run this script to initialize the virtual environment (uv is required).
```
bash setup_uv.sh
```
To activate the virtual environemnt run `source .venv/bin/activate`
Additionally, fill in the appropriate paths in the `.env` file.
```
PROJECT_ROOT="XXX/SGFM"
HYDRA_JOBS="XXX/SGFM/hydra"
WABDB_DIR="XXX/SGFM/wabdb"
WABDB_CACHE_DIR="XXX/SGFM/wabdb_cache"
```
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
Pretrained checkpoints can be found [here](https://drive.google.com/drive/folders/16Tz0LLnDPWyCkI8fAKH5FphxYR6C07ph?usp=sharing).
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

## Citation
```
@misc{puny2025spacegroupconditionalflow,
      title={Space Group Conditional Flow Matching}, 
      author={Omri Puny and Yaron Lipman and Benjamin Kurt Miller},
      year={2025},
      eprint={2509.23822},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23822}, 
}
```