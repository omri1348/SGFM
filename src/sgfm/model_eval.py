import json
import os
import pickle
import argparse
import torch
from pathlib import Path
from sgfm.common.eval_utils import load_data, load_model, sample, set_out_filename, get_gt_crystals
from sgfm.common.metrics import Crystal, RecEval, GenEval
from p_tqdm import p_map
import pandas as pd


def csp_post_sampling(pred_arr, gt_arr, args, pt_path):
    gt_crys = p_map(lambda x: Crystal(x), gt_arr)
    pred_crys = p_map(lambda x: Crystal(x), pred_arr)
    rec_evaluator = RecEval(pred_crys, gt_crys)
    recon_metrics = rec_evaluator.get_metrics(dists=True)  # sequential was better
    print("Saving results...")
    torch.save(
        {
            "eval_setting": args,
            "pred_arr": pred_arr,
            "gt_arr": gt_arr,
        },
        pt_path,
    )
    pd.DataFrame(recon_metrics).to_csv(pt_path.with_suffix(".csv"),index=False)


def csp_collecting(ranked_outdir, root_path, out_filename, args):
    all_results_files = list(ranked_outdir.glob("rank_*.pt"))
    all_results_files.sort()
    # only do this if we have results from all ranks
    if len(all_results_files) == args.num_ranks:
        print("Collecting CSP results from all ranks...")
        collector = {
            "eval_setting": args,
            "eval_settings": [],
            "pred_arr": [],
            "gt_arr": [],
        }
        csv_paths = []
        for f in all_results_files:
            result = torch.load(f)
            collector["eval_settings"].append(result["eval_setting"])
            collector["pred_arr"].extend(result["pred_arr"])
            collector["gt_arr"].extend(result["gt_arr"])

            csv_paths.append(f.with_suffix(".csv"))

        torch.save(collector, root_path / out_filename)

        # combine rms_dists, and save
        rms_dists = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
        rms_dists.to_csv(
            ranked_outdir / "rms_dists.csv",
            index=False,
        )

        # save results at ckpt level
        results = {
            "match_rate": (~rms_dists["rms_dists"].isna()).sum() / len(rms_dists),
            "mean_rms_dist": rms_dists["rms_dists"].mean(skipna=True),
            "pred_validity": rms_dists["pred_validity"].mean(skipna=True),
            "gt_validity": rms_dists["gt_validity"].mean(skipna=True),
            "joint_validity": rms_dists["joint_validity"].mean(skipna=True),
        }
        out_path_json = (root_path / out_filename).with_suffix(".json")
        with open(out_path_json, "w") as f:
            json.dump(results, f)
        print(f"Results collected and saved to {out_path_json}")
    else:
        print(f"Not all ranks have results. Expected {args.num_ranks}, but found {len(all_results_files)}.")
        print("Skipping collecting results.")


def dng_post_sampling(pred_arr, args, pt_path):
    print("Saving results...")
    pred_crys = p_map(lambda x: Crystal(x,do_dng_coverage=args.do_dng_coverage), pred_arr)
    torch.save(
        {
            "eval_setting": args,
            "pred_arr": pred_arr,
        },
        pt_path,
    )
    # Save crystal predictions as pickle file
    with open(pt_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(pred_crys, f)


def dng_collecting(model_path, ranked_outdir, root_path, out_filename, args):
    all_results_files = list(ranked_outdir.glob("rank_*.pkl"))
    all_results_files.sort()
    # only do this if we have results from all ranks
    if len(all_results_files) == args.num_ranks:
        print("Collecting DNG results from all ranks...")
        pred_crys = []
        for f in all_results_files:
            with open(f, 'rb') as pickle_file:
                result = pickle.load(pickle_file)
                pred_crys.extend(result)
        gt_crys, cfg = get_gt_crystals(model_path, args.do_dng_coverage)
        gen_evaluator = GenEval(pred_crys, gt_crys, eval_model_name=cfg.data.eval_model_name, n_samples=args.dng_num_valid_samples)
        gen_metrics = gen_evaluator.get_metrics(do_dng_coverage=args.do_dng_coverage)
        out_path_json = (root_path / out_filename).with_suffix(".json")
        with open(out_path_json, "w") as f:
            json.dump(gen_metrics, f)   
        print(f"Results collected and saved to {out_path_json}")
    else:
        print(f"Not all ranks have results. Expected {args.num_ranks}, but found {len(all_results_files)}.")
        print("Skipping collecting results.")


def set_do_eval(path: Path, overwrite: bool) -> bool:
    if path.exists():
        print("File already exists:")
        print(path)
        if overwrite:
            print("Overwriting...")
            return True
        else:
            print("Skipping evaluation...")
            return False
    else:
        return True


def main(args: argparse.Namespace):
    print("preparing paths...")
    model_path = Path(args.model_path).resolve()
    root_path = model_path.parent
    epoch = torch.load(model_path, map_location='cpu')['epoch']
    out_filename = set_out_filename(args, epoch)
    pt_path = root_path / out_filename

    rank = None
    if args.num_ranks is not None:
        # Get rank from args or environment, subtract 1 for zero-based indexing
        rank = (args.rank if args.rank is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))) - 1
        print(f"RANK {rank+1}/{args.num_ranks}")
        # Create directory for rank-specific output
        ranked_outdir = root_path / Path(pt_path).stem
        ranked_outdir.mkdir(exist_ok=True)
        pt_path = ranked_outdir / f"rank_{rank + 1:03d}.pt"

    do_eval = set_do_eval(pt_path, args.overwrite)
    print("loading...")
    model = load_model(model_path)

    if model.mode == "DNG":
        dataset_type = "train"
    else: # model type is CSP
        dataset_type = "test"
    print("dataset type is {}".format(dataset_type))
    print("model type is {}".format(model.mode))
    if do_eval:
        loader = load_data(
            root_path,
            dataset=dataset_type,
            subset_size=args.dng_data_subset_size if model.mode == "DNG" else None,
            rank=rank,
            num_ranks=args.num_ranks,
        )

        if torch.cuda.is_available():
            model.to("cuda")

        print("Evaluate the model.")
        if args.num_ranks is not None:
            print("dataset subset size", len(loader))
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
        # correct code when do_eval is False
        if model.mode == "CSP":
            csp_post_sampling(pred_arr, gt_arr, args, pt_path)
        if model.mode == "DNG":
            dng_post_sampling(pred_arr, args, pt_path)

    # only do this if we are running in a distributed setting
    if args.num_ranks is not None:
        print("Collecting results from all ranks...")
        if model.mode == "CSP":
            print("... for csp")
            csp_collecting(ranked_outdir, root_path, out_filename, args)
        elif model.mode == "DNG":
            print("... for dng")
            dng_collecting(model_path, ranked_outdir, root_path, out_filename, args)
        else:
            print("... for other modes, nothing to collect")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--label", default="")
    parser.add_argument("--num_steps", default=200, type=int)
    parser.add_argument("--slope_k", default=0, type=float)
    parser.add_argument("--slope_x", default=0, type=float)
    parser.add_argument("--dng_data_subset_size", default=10_000, type=int)
    parser.add_argument("--dng_num_valid_samples", default=1_000, type=int)
    parser.add_argument("--do_dng_coverage", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_ranks", default=None, type=int)
    parser.add_argument(
        "--rank",
        default=None,
        type=int,
        help="if not set, will use $SLURM_ARRAY_TASK_ID with 1-based indexing",
    )
    args = parser.parse_args()
    main(args)
