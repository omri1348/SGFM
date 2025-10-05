from collections import Counter
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from sgfm.common.eval_utils import smact_validity, structure_validity, compute_cov, get_coordination_numbers

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):
    def __init__(self, crys_array_dict, full_compute=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        if full_compute:
            self.get_fingerprints()
            self.get_coordination_number()



    def get_coordination_number(self):
        ase_atom = AseAtomsAdaptor.get_atoms(self.structure)
        self.cn = np.mean(get_coordination_numbers(ase_atom))

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid


    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(
        self,
        pred_crys,
        gt_crys,
        stol: float = 0.5,
        angle_tol: float = 10,
        ltol: float = 0.3,
    ):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def process_one(self, pred, gt, is_valid):
        if not is_valid:
            return None
        try:
            rms_dist = self.matcher.get_rms_dist(
                pred.structure,
                gt.structure,
            )
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return None

    def get_match_rate_and_rms(self, dists: bool):
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]
        pred_validity = np.array([c.valid for c in self.preds])
        gt_validity = np.array([c.valid for c in self.gts])

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(self.process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        out = {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist,
        }
        if dists:
            out['rms_dists'] = rms_dists
            out['pred_validity'] = pred_validity
            out['gt_validity'] = gt_validity
            out['joint_validity'] = np.logical_and(pred_validity, gt_validity)
        return out

    def get_metrics(self, dists: bool = False):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms(dists=dists))
        return metrics


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_cn_wdist(self):
        pred_cns = [c.cn for c in self.valid_samples]
        gt_cns = [c.cn for c in self.gt_crys]
        wdist_cn = wasserstein_distance(pred_cns, gt_cns)
        return {'wdist_cn': wdist_cn}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self, full_compute=False):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        if full_compute:
            metrics.update(self.get_cn_wdist())
            metrics.update(self.get_coverage())
        return metrics








