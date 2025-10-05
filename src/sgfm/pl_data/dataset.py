import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
import numpy as np
from sgfm.common.data_utils import normalize_k, preprocess, sample_x0, logmap, logmap2, preprocess_tensors
from sgfm.pl_modules.lattice.crystal_family import CrystalFamily
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch



def cryst_collate_fn(data_list):
    batch =  Batch.from_data_list(data_list,exclude_keys=['G_inv_permutation'])
    max_atoms = batch.num_atoms.max().item()
    permutation_list=[data.G_inv_permutation for data in data_list]
    pad_permutation = []
    for perm in permutation_list:
        n = perm.shape[1]
        diff_size = max_atoms - n
        if diff_size > 0:
            perm_addition = torch.arange(n, n + diff_size).unsqueeze(0).repeat(perm.shape[0], 1)
            perm = torch.cat([perm, perm_addition], dim=1)
        pad_permutation.append(perm)
    batch.G_inv_permutation = torch.cat(pad_permutation, dim=0)
    return batch
    
class CrystDataset(Dataset):
    def __init__(
            self, 
        name: ValueNode, 
        path: ValueNode,
        niggli: ValueNode, 
        primitive: ValueNode,
        graph_method: ValueNode, 
        preprocess_workers: ValueNode,
        save_path: ValueNode, 
        symprec: ValueNode,
        angle_tolerance: ValueNode,
        use_space_group: ValueNode, 
        use_pos_index: ValueNode,
        equi_sample: ValueNode, 
        mean_path: ValueNode, 
        std_path: ValueNode, 
        **kwargs
    ) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.equi_sample = equi_sample
        self.preprocess(save_path, preprocess_workers, mean_path, std_path)
        self.cf = CrystalFamily()
        print(f"Dataset size: {len(self.cached_data)}!!!!!")

    def preprocess(self, save_path, preprocess_workers, mean_path, std_path):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
            self.max_atom=100
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,  # Default angle tolerance
                use_space_group=self.use_space_group,
            )
            torch.save(cached_data, save_path)
            print(f"Saved preprocessed data to {save_path}")
            print(f"Dataset size: {len(cached_data)}")
            self.cached_data = cached_data

        if os.path.exists(mean_path) and os.path.exists(std_path):
            print(f"Loading mean and std from {mean_path} and {std_path}")
            self.mean = torch.load(mean_path).float()
            self.std = torch.load(std_path).float()
        else:
            print(f"Calculating mean and std for in {save_path}")
            normalize_k(save_path)
            self.mean = torch.load(mean_path).float()
            self.std = torch.load(std_path).float()
        print(f"Mean: {self.mean}, Std: {self.std}")

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, _, _, edge_indices,
         _, num_atoms) = data_dict['graph_arrays']
        frac_coords = torch.Tensor(frac_coords).float()
        # space group
        Point_G = data_dict['Point_G']
        G_inv_permutation = data_dict['G_inv_permutation']
        if self.equi_sample:
            x0 = sample_x0(data_dict['wyckoff_ops'])
            u = -logmap2(frac_coords, x0, data_dict['wyckoff_ops'])
            tmp_x = sample_x0(data_dict['wyckoff_ops'])
            u_mask = -logmap2(tmp_x, x0, data_dict['wyckoff_ops'])
        else: 
            x0 = torch.rand_like(frac_coords).float()
            u = -logmap(frac_coords, x0)
            u_mask = torch.zeros_like(u)
        k = torch.Tensor(data_dict['ks']).float()
        mask, bias = self.cf.get_spacegroup_constraint(data_dict['spacegroup'])
        orbit_sizes = torch.LongTensor([w.shape[0] for w in data_dict['wyckoff_ops']])
        num_orbits = torch.tensor([orbit_sizes.shape[0]]).long()
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            Point_G=Point_G,
            num_nodes=num_atoms,
            num_atoms=num_atoms,
            x0=x0,
            u=u,
            u_mask=u_mask,
            k=k.unsqueeze(0),
            k_mask=mask.unsqueeze(0),
            k_bias=bias.unsqueeze(0),
            G_inv_permutation=G_inv_permutation,
            group_sizes=torch.tensor([Point_G.shape[0]]).long(),
            k_mean=self.mean.unsqueeze(0),
            k_std=self.std.unsqueeze(0),
            orbit_sizes=orbit_sizes,
            num_orbits=num_orbits,
        )
        return data


