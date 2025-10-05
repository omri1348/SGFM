import pickle as pkl
import os
import torch
from tqdm import tqdm
from sgfm.common.utils import PROJECT_ROOT
from sgfm.common.metrics import Crystal


data_path = os.path.join(PROJECT_ROOT, "data", "mp_20")


def create_mp20_pkl(full: bool = False) -> None:
    source_path = os.path.join(data_path, "test_sym.pt")
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
        if full:
            cyrstal_arr.append(Crystal({'frac_coords': frac_coords,
                                'atom_types': atom_types,
                                'lengths': lengths,
                                'angles': angles}, 
                                full_compute=full))
        else:
            cyrstal_arr.append(Crystal({'frac_coords': frac_coords,
                                    'atom_types': atom_types,
                                    'lengths': lengths,
                                    'angles': angles}, full_compute=full))
    save_path = os.path.join(data_path, "crystal_full.pkl" if full else "crystal.pkl")
    with open(save_path, 'wb') as f:
        pkl.dump(cyrstal_arr, f)
    print(f"Saved {len(cyrstal_arr)} crystals to {save_path}")
    return


if __name__ == "__main__":
    create_mp20_pkl()
    create_mp20_pkl(full=True)