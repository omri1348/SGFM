import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Any
import hydra
import pytorch_lightning as pl
from sgfm.common.data_utils import expmap
import math

MAX_ATOMIC_NUM = 100
NUM_ATOMIC_BITS = math.floor(math.log2(MAX_ATOMIC_NUM)) + 1

def int2bits(
    x: torch.LongTensor, n: int) -> torch.Tensor:
    """Convert an integer x in (...) into bits in (..., n)."""
    x = torch.bitwise_right_shift(
        torch.unsqueeze(x, -1), torch.arange(n, dtype=x.dtype, device=x.device)
    )
    x = torch.fmod(x, 2)
    x = 2 * x - 1  # convert to {-1, 1}
    return x.float()

def bits2int(x: torch.Tensor) -> torch.Tensor:
    """Converts bits x in (..., n) into an integer in (...)."""
    signs = x.sign()
    bits = (signs + 1) / 2
    n = bits.size(-1)
    res = torch.sum(bits* (2 ** torch.arange(n).to(bits.device)), dim=-1)
    return res.long()


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

class SGFM(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if 'flow_model' in self.hparams:
            self.ut = hydra.utils.instantiate(self.hparams.flow_model, _recursive_=False)
            self.normalize_k = self.hparams.normalize_k
            self.mode = self.hparams.flow_model.mode
        else:
            self.ut = hydra.utils.instantiate(self.hparams.model.flow_model, _recursive_=False)
            self.normalize_k = self.hparams.model.normalize_k
            self.mode = self.hparams.model.flow_model.mode

    def set_xt_vt(self,x0,u,t,num_atoms):
        xt = expmap(x0, t.repeat_interleave(num_atoms, dim=0).unsqueeze(-1) * u)
        vt = u
        return xt, vt
    
    def set_kt_vkt(self,k,mask_k,t,k_mean,k_std):
        k0 = torch.randn_like(k) * mask_k
        if self.normalize_k:
            k = ((k - k_mean) / (k_std+1e-6))
        k = k*mask_k
        kt = t.unsqueeze(1)*k+(1-t.unsqueeze(1))*k0
        vkt = (k-k0) * mask_k
        return kt, vkt

    def forward(self, batch, batch_idx = None):
        group_sizes = batch.group_sizes
        group_sizes_list = group_sizes.tolist()
        #### sample time 
        t = torch.rand(group_sizes.shape[0], device=group_sizes.device)
        #### create xt and vt
        xt, vt = self.set_xt_vt(batch.x0, batch.u,t,batch.num_atoms)
        #### create kt and v_kt
        kt, vkt = self.set_kt_vkt(batch.k,batch.k_mask,t,batch.k_mean,batch.k_std)
        # atom types in bits
        if self.mode=='DNG':
            atom_types = int2bits(batch.atom_types-1, NUM_ATOMIC_BITS)
            atom_types_0 = torch.randn(batch.orbit_sizes.shape[0],NUM_ATOMIC_BITS, device=batch.x0.device)
            atom_types_0 = atom_types_0.repeat_interleave(batch.orbit_sizes, dim=0)
            t_atom_types = t.repeat_interleave(batch.num_atoms, dim=0).unsqueeze(-1)
            atom_types_t = (1-t_atom_types)*atom_types_0 + t_atom_types*atom_types
            atom_types_vt = (atom_types - atom_types_0) 
        else: 
            atom_types_t = torch.nn.functional.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
        # compute frac lattice features TODD
        lattice_data = (self.normalize_k, batch.k_mean, batch.k_std, batch.k_mask, batch.k_bias)
        # forward
        out_x, out_k, out_atom_types = self.ut(t, atom_types_t, xt, kt, batch.num_atoms, batch.batch, batch.Point_G, batch.G_inv_permutation, group_sizes_list, group_sizes, lattice_data)
        # compute loss with maks due to padding
        loss_lattice = F.mse_loss(out_k, vkt)
        loss_coord = F.mse_loss(out_x, vt)
        if self.mode == 'DNG':
            loss_atom_types = F.mse_loss(out_atom_types, atom_types_vt)
        else:
            loss_atom_types = 0
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_atom_types * loss_atom_types)
        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_atom_types' : loss_atom_types,
        }

    def sample(self,batch,num_steps,slope_k=0,slope_x=0):
        ode_opts = {"method": "euler", "options": {"step_size":1/num_steps}}
        # mask the vectorfield (change this to two random samples)
        mask_v = (batch.u_mask != 0).float()
        # sample k0
        k0 = torch.randn_like(batch.k) * batch.k_mask
        if self.mode == 'DNG':
            atom_types_0 = torch.randn(batch.orbit_sizes.shape[0],NUM_ATOMIC_BITS, device=batch.x0.device)
            atom_types_0 = atom_types_0.repeat_interleave(batch.orbit_sizes, dim=0)
            z = torch.cat([batch.x0,k0.repeat_interleave(batch.num_atoms,dim=0), atom_types_0], dim=-1)
        elif self.mode == 'CSP':
            atom_types = torch.nn.functional.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM).float()
            z = torch.cat([batch.x0,k0.repeat_interleave(batch.num_atoms,dim=0)], dim=-1)
        lattice_data = (self.normalize_k, batch.k_mean, batch.k_std, batch.k_mask, batch.k_bias)
        def ode_func(t,z):
            nonlocal atom_types
            x = z[:,:3]
            k = z[0,3:9].unsqueeze(0)
            if self.mode == 'DNG':
                a = z[:,9:]
            else:
                a = atom_types
            t = t.unsqueeze(0)
            out_x, out_k, out_atom_types = self.ut(t, a, x, k, batch.num_atoms, batch.batch, batch.Point_G, batch.G_inv_permutation, batch.group_sizes.tolist(), batch.group_sizes, lattice_data)
            if self.ut.equivariant:
                out_x = out_x * mask_v
            out_k = out_k * batch.k_mask
            if slope_x > 0:
                out_x = (1+(slope_x * t.squeeze())) * out_x
            if slope_k > 0:
                out_k = (1+(slope_k * t.squeeze())) * out_k
            if self.mode == 'DNG':
                out_z = torch.cat([out_x,out_k.repeat_interleave(batch.num_atoms,dim=0), out_atom_types], dim=-1)
            else:
                out_z = torch.cat([out_x,out_k.repeat_interleave(batch.num_atoms,dim=0)], dim=-1)
            return out_z 
        z = odeint(ode_func, z, t=torch.FloatTensor([0,1]).cuda(), **ode_opts)[-1,...]
        x_pred = z[:,:3]
        k_pred = z[0,3:9]
        if self.mode == 'DNG':
            a_pred = z[:,9:]
            a_pred = bits2int(a_pred)+1
            # clip to fit max atomic number
            if a_pred.max() > MAX_ATOMIC_NUM:
                print('non-valid crystal found, clipping atomic numbers')
            a_pred = a_pred.clip(max=MAX_ATOMIC_NUM)
        else:
            a_pred = None
        # normalizing and projecting
        if self.normalize_k:
            k_pred = (k_pred * batch.k_std) + batch.k_mean
        k_pred = k_pred * batch.k_mask + batch.k_bias
        
        return x_pred, k_pred, a_pred
    
    def on_after_backward(self):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        total_norm = 0.
        for nm,p in self.ut.named_parameters():
            try:
                param_norm = p.grad.data.norm(2)
                total_norm = total_norm + param_norm.item() ** 2
            except:
                pass
        total_norm = total_norm ** (1. / 2)

        self.log_dict({
            'grad_norm': total_norm
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        # torch.cuda.empty_cache()
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch, batch_idx)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom_types = output_dict['loss_atom_types']
        loss = output_dict['loss']

        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'atom_types_loss': loss_atom_types,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        if loss.isnan() or loss.isinf():
            print(batch_idx)
            return None

        return loss


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return 0

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return 0

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_atom_types = output_dict['loss_atom_types']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_atom_types_loss': loss_atom_types,
        }

        return log_dict, loss
    
