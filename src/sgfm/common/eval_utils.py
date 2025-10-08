import os
import itertools
from typing import Literal, Optional, List, Dict
import numpy as np
import torch
import hydra
from ase import Atoms
from ase.data import covalent_radii


from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra import compose, initialize_config_dir
from pathlib import Path

import smact
from smact.screening import pauling_test
from torch_geometric.data import Batch

import pickle as pkl
from sgfm.pl_modules.sgfm_model import SGFM
import torch.utils

from sgfm.common.constants import CompScalerMeans, CompScalerStds
from sgfm.common.data_utils import StandardScaler, chemical_symbols
from sgfm.common.utils import PROJECT_ROOT
from sgfm.pl_modules.lattice.crystal_family import CrystalFamily
from sgfm.pl_modules.sgfm_model import MAX_ATOMIC_NUM

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

def set_out_filename(args, epoch):
    if args.label == "":
        out_filename = f"eval_diff_epoch{epoch:05d}_steps{args.num_steps:04d}_sk{args.slope_k:0.3f}_sx_{args.slope_x:0.3f}.pt"  
    else:
        out_filename = f"eval_diff_epoch{epoch:05d}_steps{args.num_steps:04d}_sk{args.slope_k:0.3f}_sx_{args.slope_x:0.3f}_{args.label}.pt"
    if args.full_compute:
        out_filename = f"full_{out_filename}"
    return out_filename


def load_model(model_path: Path) -> torch.nn.Module:
    root_path = str(model_path.parent)
    hparams = os.path.join(root_path, "hparams.yaml")
    print('loading model from checkpoint..')
    model = SGFM.load_from_checkpoint(str(model_path), hparams_file=hparams, strict=True)
    print('done loading model')
    return model

def load_data(
    root_path,
    dataset: Literal["train", "val", "test"] = "test",
    subset_size: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    with initialize_config_dir(str(root_path), version_base="1.1"):
        cfg = compose(config_name='hparams')
        print('loading data...')
        cfg.data.datamodule.batch_size.test = 1
        cfg.data.datamodule.batch_size.train = 1
        datamodule = hydra.utils.instantiate(
            cfg.data.datamodule,
            _recursive_=False,
        )
        subset_inds = None
        if dataset == "train":
            print('Loading train data')
            datamodule.setup('fit')
            subset_inds = np.random.choice(len(datamodule.train_dataset), subset_size)
            loader = datamodule.train_dataloader(shuffle=False, subset_inds=subset_inds)
        elif dataset == "test":
            print('Loading test data')
            datamodule.setup("test")
            loader = datamodule.test_dataloader(subset_inds=subset_inds)[0]

    return loader


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles

def update_angles_length(d, k, crystal_family):
    lattice = crystal_family.v2m(k.float())
    lengths, angles = lattices_to_params_shape(lattice)
    d["lengths"] = lengths.squeeze().detach().cpu().numpy()
    d["angles"] = angles.squeeze().detach().cpu().numpy()
    return d

def update_frac_coords(d,frac_coords):
    """
    Update the fractional coordinates in the dictionary `d` with the provided `frac_coords`.
    """
    d["frac_coords"] = frac_coords.squeeze().detach().cpu().numpy()
    if len(d["frac_coords"].shape) == 1:
        d["frac_coords"] = np.expand_dims(d["frac_coords"], axis=0)
    return d

def update_atom_types(gt, pred, atom_types, out_a, batch):
    if out_a is not None:
        atom_types = out_a
    if atom_types.ndim > 1 and atom_types.shape[-1] == MAX_ATOMIC_NUM:
        # If atom types are one hot encoded
        atom_types = atom_types.argmax(dim=-1).squeeze().detach().cpu().numpy() + 1
    else:
        # If atom types are already indices
        atom_types = atom_types.squeeze().detach().cpu().numpy()
    pred["atom_types"] = atom_types
    if len(pred["atom_types"].shape) == 0:
        pred["atom_types"] = np.array([pred["atom_types"]])
    if out_a is not None:
        gt["atom_types"] = batch.atom_types.squeeze().detach().cpu().numpy()
    else:
        gt["atom_types"] = atom_types
    if len(gt["atom_types"].shape) == 0:
        gt["atom_types"] = np.array([gt["atom_types"]])
    return gt, pred


def sample(loader, model, num_steps=1000, verbose=False, slope_k=0, slope_x=0):
    crystal_family = CrystalFamily()
    pred_arr = []
    gt_arr = []
    for idx, batch in enumerate(loader):
        gt = {}
        pred = {}
        if isinstance(batch, Batch):
            batch = batch.cuda()
            frac_coords = batch.frac_coords
            k = batch.k
            atom_types = batch.atom_types
            Point_G = batch.Point_G
        print("batch ", idx, "x shape", frac_coords.shape[0], "G shape", Point_G.shape[0])
        out_frac, out_k, out_a = model.sample(batch, num_steps, slope_k=slope_k, slope_x=slope_x)
        # update frac_coords
        gt = update_frac_coords(gt, frac_coords%1)
        pred = update_frac_coords(pred, out_frac%1)
        # extract and save lattice parameters
        gt = update_angles_length(gt, k, crystal_family)
        pred = update_angles_length(pred, out_k, crystal_family)
        # save atom types
        gt, pred = update_atom_types(gt, pred, atom_types, out_a, batch)
        gt_arr.append(gt)
        pred_arr.append(pred)
    return pred_arr, gt_arr


def get_gt_crystals(model_path, args):
    with initialize_config_dir(str(model_path.parent), version_base="1.1"):
        cfg = compose(config_name='hparams')
    if args.full_cmpute:
        test_file_path = Path(cfg.data.root_path) / "crystal_full.pkl"
    else:
        test_file_path = Path(cfg.data.root_path) / "crystal.pkl"
    print(f"Loading crystals from {test_file_path}")
    crystal_arr = pkl.load(open(test_file_path, 'rb'))
    return crystal_arr


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


def get_bonds(atoms: Atoms, covalent_increase_factor: float = 1.25) -> List[List[int]]:
    """
    Compute the list of bonds for every atom in the given structure.

    The bonds are determined by comparing the distances between the atoms and the involved covalent radii. A bond is
    present if the two atoms are closer than the sum of their covalent radii times a given multiplicative factor.

    :param atoms:
        The structure to calculate the bonds for.
    :type atoms: Atoms
    :param covalent_increase_factor:
        The factor by which to multiply the sum of the covalent radii to determine the bond distance.
        Defaults to 1.25.
    :type covalent_increase_factor: float

    :return:
        List of bonds for the atoms in the structure.
    :rtype: List[List[int]]
    """
    distances = atoms.get_all_distances(mic=True)
    cr = [covalent_radii[number] for number in atoms.numbers]

    # List of bonded atoms for every atom.
    bonds = [[] for _ in range(len(atoms))]

    for first_index in range(len(atoms)):
        for second_index in range(first_index + 1, len(atoms)):
            if (cr[first_index] + cr[second_index]) * covalent_increase_factor >= distances[first_index, second_index]:
                bonds[first_index].append(second_index)
                bonds[second_index].append(first_index)

    return bonds


def get_coordination_numbers(atoms: Atoms, covalent_increase_factor: float = 1.25) -> List[int]:
    """
    Compute the coordination numbers for the atoms in the given structure.

    The coordination numbers are determined by comparing the distances between the atoms and the involved covalent
    radii. A bond that increases the coordination number is present if the two atoms are closer than the sum of their
    covalent radii times a given multiplicative factor.

    :param atoms:
        The structure to calculate the coordination numbers for.
    :type atoms: Atoms
    :param covalent_increase_factor:
        The factor by which to multiply the sum of the covalent radii to determine the bond distance.
        Defaults to 1.25.
    :type covalent_increase_factor: float

    :return:
        List of coordination numbers for the atoms in the structure.
    :rtype: List[int]
    """
    bonds = get_bonds(atoms, covalent_increase_factor)
    return [len(b) for b in bonds]