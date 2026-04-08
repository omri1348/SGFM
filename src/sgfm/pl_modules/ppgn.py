import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.utils import to_dense_batch
from sgfm.pl_modules.lattice.crystal_family import CrystalFamily


def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = 2 * math.pi * (y - x)
    return torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)


def timestep_embedding(timesteps, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=32, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        if len(x.shape) == 3:
            emb = emb.reshape(
                x.shape[0], x.shape[1], self.n_frequencies * self.n_space
            )
        elif len(x.shape) == 4:
            emb = emb.reshape(
                x.shape[0],
                x.shape[1],
                x.shape[2],
                self.n_frequencies * self.n_space,
            )
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MlpBlock(nn.Module):
    def __init__(self, width, activation_fn=nn.functional.silu):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.activation = activation_fn
        self.linear2 = nn.Linear(width, width)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.activation(out)
        out = self.linear2(out)
        return out


class PPGN_block(nn.Module):
    """
    Inputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result,
    and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth.
    """

    def __init__(self, width):
        super().__init__()
        self.mlp1 = MlpBlock(width)
        self.mlp2 = MlpBlock(width)
        self.ln = nn.LayerNorm(width)

    def forward(self, inputs, mask):
        mlp1 = self.mlp1(inputs) * mask
        mlp2 = self.mlp2(inputs) * mask
        out = self.ln(
            torch.einsum("bikl,bkjl->bijl", mlp1, mlp2)
        ) + inputs  # B x N x N x d
        return out


class PPGN(nn.Module):
    def __init__(
        self,
        layers,
        width,
        pos_dim,
        k_dim,
        atom_dim,
        n_frequencies=128,
        t_frequencies=128,
    ):
        """
        Build the model computation graph, until scores/values are returned
        at the end.
        """
        super().__init__()

        depth = layers
        self.dif_embed = SinusoidsEmbedding(
            n_frequencies=n_frequencies, n_space=pos_dim
        )
        self.t_dim = t_frequencies
        self.reg_blocks = nn.ModuleList()
        self.feature_embed_arr = nn.ModuleList()
        embed_dim = atom_dim + self.t_dim
        self.feature_embed = nn.Linear(embed_dim, width)
        self.cf = CrystalFamily()
        for _ in range(depth):
            self.feature_embed_arr.append(
                nn.Sequential(
                    nn.Linear(
                        width + k_dim + pos_dim + self.dif_embed.dim, width
                    ),
                    nn.SiLU(),
                )
            )
            self.reg_blocks.append(PPGN_block(width))

    def compute_frac_lattice_features(self, k, lattice_data, frac_diff, quad_mask):
        normalize_k, k_mean, k_std, k_mask, k_bias = lattice_data
        y = k.clone()
        if normalize_k:
            y = (y * k_std) + k_mean
        y = (y * k_mask) + k_bias
        lattices_mat = self.cf.v2m(y)
        ltl = torch.matmul(lattices_mat.transpose(-1, -2), lattices_mat)
        ltl_f = torch.einsum("bij,bklj->bkli", ltl, frac_diff)
        l_f_features = ltl_f / (ltl_f.norm(dim=-1, keepdim=True) + 1e-6)
        return l_f_features * quad_mask

    def forward(self, frac_coords, k, atom_types, t, mask, lattice_data, num_atoms):
        b, n, d = frac_coords.shape
        n_arr = torch.arange(n).to(frac_coords.device)
        quad_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        frac_diff_raw = logmap(
            frac_coords.unsqueeze(2), frac_coords.unsqueeze(1)
        )  # b x n x n x 3
        frac_diff = self.dif_embed(frac_diff_raw)  # b x n x n x embed_dim

        t_embed = timestep_embedding(t, self.t_dim)  # b x 128
        t_embed = t_embed.unsqueeze(1).repeat(1, n, 1)  # b x n x 128
        k_expand = k.unsqueeze(1).unsqueeze(1).repeat(
            1, n, n, 1
        )  # b x n x n x 6

        embed = torch.cat([atom_types, t_embed], dim=-1)
        embed = (
            self.feature_embed(embed)
            .transpose(1, 2)
            .diag_embed(dim1=1, dim2=2)
        )  # b x n x n x width

        frac_lattice_features = self.compute_frac_lattice_features(
            k, lattice_data, frac_diff_raw, quad_mask
        )  # b x n x n x 3

        for i, block in enumerate(self.reg_blocks):
            embed = self.feature_embed_arr[i](
                torch.cat(
                    [embed, frac_lattice_features, frac_diff, k_expand], dim=-1
                )
            )  # b x n x n x width
            embed = embed * quad_mask
            embed = block(embed, quad_mask)

        # extract node features
        diag = embed[:, n_arr.long(), n_arr.long(), :]
        return diag * mask


class GroupAveragedPPGN(nn.Module):
    def __init__(
        self,
        width=256,
        layers=6,
        pos_dim=3,
        k_dim=6,
        atom_dim=100,
        equivariant=True,
        n_frequencies=128,
        t_frequencies=128,
        atom_type_dim=0,
    ):
        super(GroupAveragedPPGN, self).__init__()
        self.ppgn = PPGN(
            layers,
            width,
            pos_dim,
            k_dim,
            atom_dim,
            n_frequencies,
            t_frequencies,
        )
        self.width = width
        self.output_x = nn.Linear(width, pos_dim)
        self.output_k = nn.Linear(width, k_dim)
        self.equivariant = equivariant
        self.atom_type_dim = atom_type_dim
        if atom_type_dim > 0:
            self.output_atom_types = nn.Linear(width, atom_type_dim)

    def forward(
        self,
        x,
        ks,
        a,
        t,
        mask,
        lattice_data,
        num_atoms,
        G,
        group_size,
        tensor_group_size,
        inv_G_permutation,
    ):
        out = self.ppgn(
            x, ks, a, t, mask, lattice_data, num_atoms
        )  # B x N x d
        out_x = self.output_x(out)  # B x N x 3
        out_k = out.sum(dim=1) / num_atoms.view(-1, 1)
        out_k = self.output_k(out_k)  # B x 6
        if self.atom_type_dim > 0:
            out_atom_types = self.output_atom_types(out)  # B x N x atom_type_dim
            if self.equivariant:
                out_atom_types = self.invariant_split_and_average(
                    out_atom_types, inv_G_permutation, group_size, tensor_group_size
                )
            out_atom_types = out_atom_types * mask
        else:
            out_atom_types = None
        if self.equivariant:
            out_x = self.split_and_average_x(
                out_x, G, inv_G_permutation, group_size, tensor_group_size
            )
        out_x = out_x * mask
        return out_x, out_k, out_atom_types

    def split_and_average_x(
        self, x, G, inv_G_permutation, group_size, tensor_group_size
    ):
        x = torch.repeat_interleave(x, tensor_group_size, dim=0)
        batch_indices = (
            torch.arange(x.size(0)).unsqueeze(1).expand_as(inv_G_permutation)
        )
        x = x[batch_indices, inv_G_permutation]
        x = torch.cat(
            [x, torch.ones(x.shape[0], x.shape[1], 1).float().to(x.device)],
            dim=-1,
        )
        z = torch.matmul(x, G.transpose(-1, -2))[:, :, :3]
        z_arr = torch.split(z, group_size, dim=0)
        z = torch.cat(
            [tmp_z.mean(dim=0, keepdim=True) for tmp_z in z_arr], dim=0
        )
        return z

    def invariant_split_and_average(
        self, x, inv_G_permutation, group_size, tensor_group_size
    ):
        x = torch.repeat_interleave(x, tensor_group_size, dim=0)
        batch_indices = (
            torch.arange(x.size(0)).unsqueeze(1).expand_as(inv_G_permutation)
        )
        x = x[batch_indices, inv_G_permutation]
        z_arr = torch.split(x, group_size, dim=0)
        z = torch.cat(
            [tmp_z.mean(dim=0, keepdim=True) for tmp_z in z_arr], dim=0
        )
        return z


class PPGNFlowModel(nn.Module):
    """Wrapper that adapts GroupAveragedPPGN to match the SGFMNet interface.

    SGFMNet.forward signature:
        forward(t, atom_types, frac_coords, lattices, num_atoms, node2graph,
                G, inv_G_permutation, group_size, tensor_group_size, lattice_data)
        -> (out_x, out_k, out_atom_types)

    This class converts between the sparse (torch_geometric) representation
    used by SGFMNet and the dense (B, N_max, ...) representation used by PPGN.
    """

    def __init__(
        self,
        mode,
        width,
        num_layers,
        atom_dim,
        n_frequencies,

        atom_type_dim=0,
        **kwargs,
    ):
        super(PPGNFlowModel, self).__init__()

        if mode not in ("CSP", "DNG"):
            raise ValueError(
                f"PPGNFlowModel supports mode='CSP' or 'DNG', got '{mode}'"
            )

        self.mode = mode
        self.equivariant = True
        self.atom_dim = atom_dim
        self.atom_type_dim = atom_type_dim


        self.ga_ppgn = GroupAveragedPPGN(
            width=width,
            layers=num_layers,
            pos_dim=3,
            k_dim=6,
            atom_dim=atom_dim,
            equivariant=True,
            n_frequencies=n_frequencies,
            t_frequencies=128,
            atom_type_dim=atom_type_dim,
        )

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        G,
        inv_G_permutation,
        group_size,
        tensor_group_size,
        lattice_data,
    ):
        # --- 1. Convert sparse -> dense ---

        # frac_coords: [total_atoms, 3] -> [B, N_max, 3]
        frac_coords_dense, mask_bool = to_dense_batch(
            frac_coords, node2graph
        )  # mask_bool: [B, N_max]
        n_max = frac_coords_dense.shape[1]

        # atom_types: [total_atoms] (CSP, integer) or [total_atoms, D] (DNG, float)
        # Convert to dense representation for PPGN.
        if atom_types.dim() == 1:
            atom_types_onehot = F.one_hot(
                atom_types.long(), num_classes=self.atom_dim
            ).float()
        else:
            atom_types_onehot = atom_types

        atom_types_dense, _ = to_dense_batch(
            atom_types_onehot, node2graph
        )  # [B, N_max, atom_dim]

        # mask: [B, N_max] -> [B, N_max, 1] for broadcasting
        mask = mask_bool.unsqueeze(-1).float()

        # lattices: already [B, 6]
        # t: already [B]

        # --- 2. Call GroupAveragedPPGN ---
        out_x_dense, out_k, out_atom_types_dense = self.ga_ppgn(
            frac_coords_dense,
            lattices,
            atom_types_dense,
            t,
            mask,
            lattice_data,
            num_atoms,
            G,
            group_size,
            tensor_group_size,
            inv_G_permutation,
        )

        # --- 3. Convert dense -> sparse ---
        # out_x_dense: [B, N_max, 3] -> [total_atoms, 3]
        out_x = out_x_dense[mask_bool]

        # out_atom_types_dense: [B, N_max, atom_type_dim] or None
        if out_atom_types_dense is not None:
            out_atom_types = out_atom_types_dense[mask_bool]
        else:
            out_atom_types = None

        return out_x, out_k, out_atom_types
