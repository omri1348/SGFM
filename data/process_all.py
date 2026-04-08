"""Process all datasets from CSV to parquet format.

Usage:
    python data/process_all.py           # Process all datasets
    python data/process_all.py mp_20     # Process just mp_20
"""
from argparse import ArgumentParser
from pathlib import Path
import time

from sgfm.common.data_utils import preprocess
from sgfm.common.parquet_utils import save_parquet

DATA_DIR = Path(__file__).resolve().parent

DATASETS = {
    "mp_20": DATA_DIR / "mp_20",
    "mpts_52": DATA_DIR / "mpts_52",
    "carbon_24": DATA_DIR / "carbon_24",
    "perov_5": DATA_DIR / "perov_5",
    "alex_mp_20": DATA_DIR / "alex_mp_20",
}

SPLITS = ["train", "val", "test"]


def process_dataset(name: str, directory: Path, num_workers: int = 94, overwrite: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Processing dataset: {name} ({directory})")
    print(f"{'='*60}")

    for split in SPLITS:
        csv_path = directory / f"{split}.csv"
        parquet_path = directory / f"{split}_sym.parquet"

        if not csv_path.exists():
            print(f"  Skipping {split}: {csv_path} not found")
            continue

        if parquet_path.exists() and not overwrite:
            print(f"  Skipping {split}: {parquet_path} already exists")
            continue

        print(f"  Processing {split}...")
        t0 = time.time()
        cached_data = preprocess(
            str(csv_path),
            num_workers=num_workers,
            niggli=True,
            primitive=False,
            graph_method="crystalnn",
            symprec=0.1,
            angle_tolerance=5,
            use_space_group=True,
        )
        save_parquet(cached_data, parquet_path)
        elapsed = time.time() - t0
        print(f"  Done {split}: {len(cached_data)} records in {elapsed:.1f}s -> {parquet_path}")


def main():
    parser = ArgumentParser(description="Process crystal datasets to parquet")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=list(DATASETS.keys()) + [[]],
        default=[],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=94,
        help="Number of parallel workers (default: 94)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
    )
    args = parser.parse_args()

    targets = args.datasets if args.datasets else list(DATASETS.keys())

    for name in targets:
        directory = DATASETS[name]
        if not directory.exists():
            print(f"Skipping {name}: directory {directory} not found")
            continue
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            print(f"WARNING: Skipping {name}: no CSV files found in {directory}. "
                  f"You must provide the CSV files yourself (see data/{name}/README.md).")
            continue
        process_dataset(name, directory, num_workers=args.num_workers, overwrite=args.overwrite)

    print("\nAll done!")


if __name__ == "__main__":
    main()
