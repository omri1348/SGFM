import torch
import torch.nn as nn

import math
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_batch
from sgfm.pl_modules.lattice.crystal_family import CrystalFamily

from sgfm.pl_modules.sgfm_model import NUM_ATOMIC_BITS


def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = 2 * math.pi * (y - x)
    return torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def timestep_embedding(timesteps, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim + 3, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattice_rep, edge_index, edge2graph, frac_diff=None, l_f_features=None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        lattice_rep_edges = lattice_rep[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_rep_edges, frac_diff, l_f_features], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None, l_f_features=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff, l_f_features)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class SGFMNet(nn.Module):

    def __init__(
        self,
        hidden_dim = 128,
        num_layers = 4,
        atom_dim = NUM_ATOMIC_BITS,
        t_frequency = 128,
        n_frequency = 10,
        ln = False,
        dense = False,
        ip = True,
        pooling='mean',
        equivariant=True,
        mode='CSP'
    ):
        super(SGFMNet, self).__init__()

        self.ip = ip
        self.mode = mode
        self.node_embedding = nn.Linear(atom_dim, hidden_dim)
        self.cf = CrystalFamily()
        
        self.atom_latent_emb = nn.Linear(hidden_dim + t_frequency, hidden_dim)
        self.act_fn = nn.SiLU()
        self.t_dim = t_frequency
        self.dis_emb = SinusoidsEmbedding(n_frequencies=n_frequency)
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )            
        self.num_layers = num_layers

        self.dense = dense

        hidden_dim_before_out = hidden_dim

        if self.dense:
            hidden_dim_before_out = hidden_dim_before_out * (num_layers + 1)

        self.coord_out = nn.Linear(hidden_dim_before_out, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim_before_out, 6, bias = False)
        if self.mode == 'DNG':
            self.atom_out = nn.Linear(hidden_dim_before_out, NUM_ATOMIC_BITS, bias = False)

        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)



        self.pooling = pooling
        self.equivariant = equivariant

    def gen_edges(self, num_atoms, frac_coords):
        # maybe can be replaced wth edge_index
        lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, logmap(frac_coords[fc_edges[0]], frac_coords[fc_edges[1]])
            
    def compute_frac_lattice_features(self, lattice, lattice_data, frac_diff, edge2graph):
        # lattices (b,6)
        # frac_diff (e,3)
        normalize_k, k_mean, k_std, k_mask, k_bias = lattice_data
        y = lattice.clone()
        if normalize_k:
            y = (y * k_std) + k_mean
        y = (y * k_mask) + k_bias
        lattices_mat = self.cf.v2m(y)
        # lattices_mat (b,3,3)
        ltl = torch.matmul(lattices_mat.transpose(-1,-2),lattices_mat)
        ltl_f = torch.einsum('eij,ej->ei', ltl[edge2graph], frac_diff)
        l_f_features = ltl_f/(ltl_f.norm(dim=-1, keepdim=True)+1e-6)
        return l_f_features

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph, G, inv_G_permutation, group_size, tensor_group_size, lattice_data):

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]
        l_f_features = self.compute_frac_lattice_features(lattices, lattice_data, frac_diff, edge2graph)
        node_features = self.node_embedding(atom_types)
        t_embed = timestep_embedding(t, self.t_dim)
        t_per_atom = t_embed.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff,l_f_features=l_f_features)
            if i != self.num_layers - 1:
                h_list.append(node_features)
        if self.ln:
            node_features = self.final_layer_norm(node_features)

        h_list.append(node_features)

        if self.dense:
            node_features = torch.cat(h_list, dim = -1)
        graph_features = scatter(node_features, node2graph, dim = 0, reduce = self.pooling)


        coord_out = self.coord_out(node_features)
        if self.equivariant:
            out_x = self.equivariant_split_and_average_x(coord_out, G, inv_G_permutation, group_size, tensor_group_size, node2graph, num_atoms)
        else:
            out_x = coord_out
        lattice_out = self.lattice_out(graph_features)
        if self.mode == 'DNG':
            atom_out = self.atom_out(node_features)
            if self.equivariant:
                out_atom_types = self.invariant_split_and_average_x(atom_out, G, inv_G_permutation, group_size, tensor_group_size, node2graph, num_atoms)
            else: 
                out_atom_types = atom_out
        else:
            out_atom_types = None

        return out_x, lattice_out, out_atom_types
    
    def invariant_split_and_average_x(self, node_features, G, inv_G_permutation, group_size, tensor_group_size, node2graph, num_atoms):
        x, _ = to_dense_batch(node_features, node2graph)
        x = x.repeat_interleave(tensor_group_size, dim=0)
        # applying permutation
        batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(inv_G_permutation)
        x = x[batch_indices, inv_G_permutation]
        z_arr = torch.split(x, group_size, dim=0)
        z = torch.cat([tmp_z.mean(dim=0)[:num_atoms[i],:] for i,tmp_z in enumerate(z_arr)], dim=0)
        return z
    
    def equivariant_split_and_average_x(self, node_features, G, inv_G_permutation, group_size, tensor_group_size, node2graph, num_atoms):
        x, _ = to_dense_batch(node_features, node2graph)
        x = x.repeat_interleave(tensor_group_size, dim=0)
        # applying permutation
        batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(inv_G_permutation)
        x = x[batch_indices, inv_G_permutation]
        # applying point group
        x = torch.cat([x,torch.ones(x.shape[0],x.shape[1],1).float().to(x.device)],dim=-1)
        z = torch.matmul(x, G.transpose(-1,-2))[:,:,:3]
        # averaging and returning to batch format
        z_arr = torch.split(z, group_size, dim=0)
        z = torch.cat([tmp_z.mean(dim=0)[:num_atoms[i],:] for i,tmp_z in enumerate(z_arr)], dim=0)
        return z