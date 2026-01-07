from argparse import ArgumentParser
import pickle as pkl
import os
import torch
from tqdm import tqdm
from sgfm.common.utils import PROJECT_ROOT
from sgfm.common.metrics import Crystal


def create_pkl(root: str, do_dng_coverage: bool = False) -> None:
    source_path = os.path.join(root, "test_sym.pt")
    source_data = torch.load(source_path)
    cyrstal_arr = []
    for i, d in tqdm(enumerate(source_data)):
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            _,
            _,
            _,
        ) = d['graph_arrays']
        cyrstal_arr.append(Crystal({'frac_coords': frac_coords,
                            'atom_types': atom_types,
                            'lengths': lengths,
                            'angles': angles}, 
                            do_dng_coverage=do_dng_coverage))
    save_path = os.path.join(root, "crystal_full.pkl" if do_dng_coverage else "crystal.pkl")
    with open(save_path, 'wb') as f:
        pkl.dump(cyrstal_arr, f)
    print(f"Saved {len(cyrstal_arr)} crystals to {save_path}")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", type=str, choices=["mp_20", "alex_mp_20"])
    args = parser.parse_args()

    data_path = os.path.join(PROJECT_ROOT, "data", args.data)
    create_pkl(data_path)
    create_pkl(data_path, do_dng_coverage=True)
