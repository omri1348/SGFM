import json
import time
import argparse
import torch
from pathlib import Path
from sgfm.common.eval_utils import load_data, load_model, sample, set_out_filename, get_gt_crystals
from sgfm.common.metrics import Crystal, RecEval, GenEval
from p_tqdm import p_map

def csp_post_sampling(pred_arr, gt_arr,args, pt_path):
    gt_crys = p_map(lambda x: Crystal(x), gt_arr)
    pred_crys = p_map(lambda x: Crystal(x), pred_arr)
    rec_evaluator = RecEval(pred_crys, gt_crys)
    recon_metrics = rec_evaluator.get_metrics(dists=False)  # sequential was better
    print("Saving results...")
    torch.save(
        {
            "eval_setting": args,
            "pred_arr": pred_arr,
            "gt_arr": gt_arr,
        },
        pt_path,
    )
    out_path_json = pt_path.with_suffix(".json")
    with open(out_path_json, "w") as f:
        json.dump(recon_metrics, f)
    print(f"Results collected and saved to {out_path_json}")


def dng_post_sampling(pred_arr, args, pt_path, model_path):
    print("Saving results...")
    pred_crys = p_map(lambda x: Crystal(x,full_compute=args.full_compute), pred_arr)
    torch.save(
        {
            "eval_setting": args,
            "pred_arr": pred_arr,
        },
        pt_path)
    gt_crys = get_gt_crystals(model_path, args)
    gen_evaluator = GenEval(pred_crys, gt_crys, eval_model_name=cfg.data.eval_model_name, n_samples=args.dng_num_valid_samples)
    gen_metrics = gen_evaluator.get_metrics(do_coverage=args.dng_compute_coverage)
    out_path_json = pt_path.with_suffix(".json")
    with open(out_path_json, "w") as f:
        json.dump(gen_metrics, f)   
    print(f"Results collected and saved to {out_path_json}")
    


def main(args: argparse.Namespace):
    print("preparing paths...")
    model_path = Path(args.model_path).resolve()
    root_path = model_path.parent
    epoch = torch.load(model_path, map_location='cpu')['epoch']
    out_filename = set_out_filename(args, epoch)
    pt_path = root_path / out_filename

    print("loading...")
    model = load_model(model_path)
    if torch.cuda.is_available():
        model.to("cuda")

    if model.mode == "DNG":
        dataset_type = "train"
    else: # model type is CSP
        dataset_type = "test"
    print("dataset type is {}".format(dataset_type))
    print("model type is {}".format(model.mode))
    loader = load_data(
        root_path,
        dataset=dataset_type,
        subset_size=args.dng_data_subset_size if model.mode == "DNG" else None,
    )
        
    print("Evaluate the model.")
    start_time = time.time()
    print("num_steps", args.num_steps)
    print("slope_k", args.slope_k)
    print("slope_x", args.slope_x)
    with torch.inference_mode():
        print("Sampling...")
        pred_arr, gt_arr = sample(
            loader,
            model,
            num_steps=args.num_steps,
            slope_k=args.slope_k,
            slope_x=args.slope_x,
        )

    if model.mode == "CSP":
        csp_post_sampling(pred_arr, gt_arr, args, pt_path)
    if model.mode == "DNG":
        dng_post_sampling(pred_arr, gt_arr, start_time, args, pt_path, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--label", default="")
    parser.add_argument("--num_steps", default=200, type=int)
    parser.add_argument("--slope_k", default=0, type=float)
    parser.add_argument("--slope_x", default=0, type=float)
    parser.add_argument("--dng_data_subset_size", default=10_000, type=int)
    parser.add_argument("--dng_num_valid_samples", default=1_000, type=int)
    parser.add_argument("--full_compute", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
