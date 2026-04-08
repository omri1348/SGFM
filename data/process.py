from argparse import ArgumentParser, Namespace
from pathlib import Path

from sgfm.common.data_utils import preprocess
from sgfm.common.parquet_utils import save_parquet


def main(args: Namespace) -> None:
    keys = ["train", "val", "test"]
    csvs = {k: str(args.directory / f"{k}.csv") for k in keys}

    for key in keys:
        print("working on", csvs[key])
        cached_data = preprocess(
            csvs[key],
            num_workers=94,
            niggli=True,
            primitive=False,
            graph_method="crystalnn",
            symprec=0.1,
            angle_tolerance=5,
            use_space_group=True,
        )
        save_path = args.directory / f"{key}_sym.parquet"
        save_parquet(cached_data, save_path)
        print("done with", csvs[key])


if __name__ == "__main__":
    parser = ArgumentParser(description="Process crystal data")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the CSV files to process.",
    )
    args = parser.parse_args()

    main(args)
