"""GCN/GAT with dense adjacent matrix implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)


    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats

class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats
    
class GNNModel(nn.Module):
    def __init__(
        self, 
        c_in, 
        c_hidden, 
        c_out, 
        num_layers=2, 
        layer_name="GCN", 
        dp_rate=0.0, 
        num_heads=3, 
        alpha=0.2):
        """N-layer graph neural network"""
        super().__init__()
        assert layer_name in ("GAT", "GCN")
        layers = []
        in_channels, out_channels = c_in, c_hidden
        if layer_name == "GAT":
            for l_idx in range(num_layers-1):
                layers += [
                    GATLayer(c_in=in_channels,
                             c_out=out_channels,
                             num_heads=num_heads, 
                             alpha=alpha),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dp_rate)
                ]
                in_channels = c_hidden
            layers += [GATLayer(c_in=in_channels,
                                c_out=c_out,
                                num_heads=num_heads, 
                                alpha=alpha)]

        elif layer_name == "GCN":
            for l_idx in range(num_layers-1):
                layers += [
                    GCNLayer(c_in=in_channels,
                             c_out=out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dp_rate)
                ]
                in_channels = c_hidden
            layers += [GCNLayer(c_in=in_channels,
                                c_out=c_out)]           

        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj_matrix):
        """
        Inputs:
            x - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, GCNLayer) or isinstance(l, GATLayer):
                x = l(x, adj_matrix)
            else:
                x = l(x)
        return x

class PropNet(nn.Module):
    """Interaction Network in: https://github.com/pairlab/v-cdn/blob/master/model_modules.py"""
    
    def __init__(self, node_dim_in, edge_dim_in, nf_hidden, node_dim_out, edge_dim_out,
                 edge_type_num=1, pstep=2, batch_norm=1, use_gpu=True):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.nf_hidden = nf_hidden

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.edge_type_num = edge_type_num
        self.pstep = pstep

        # node encoder
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        self.edge_encoders = nn.ModuleList()
        for i in range(edge_type_num):
            modules = [
                nn.Linear(node_dim_in * 2 + edge_dim_in, nf_hidden),
                nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(nf_hidden))

            self.edge_encoders.append(nn.Sequential(*modules))

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(nf_hidden * 3, nf_hidden),
            nn.ReLU(),
            nn.Linear(nf_hidden, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList()
            for j in range(edge_type_num):
                modules = [
                    # input: node_rep * 2, edge_enc, edge_rep
                    nn.Linear(nf_hidden * 3, nf_hidden),
                    nn.ReLU(),
                    nn.Linear(nf_hidden, nf_hidden),
                    nn.ReLU()]
                if batch_norm == 1:
                    modules.append(nn.BatchNorm1d(nf_hidden))
                edge_propagator.append(nn.Sequential(*modules))

            self.edge_propagators.append(edge_propagator)

        # node predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, node_dim_out))
        self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(self, node_rep, edge_rep=None, edge_type=None, start_idx=0,
                ignore_node=False, ignore_edge=False):
        # node_rep: B x N x node_dim_in
        # edge_rep: B x N x N x edge_dim_in
        # edge_type: B x N x N x edge_type_num
        # start_idx: whether to ignore the first edge type
        B, N, _ = node_rep.size()

        # node_enc
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(B, N, self.nf_hidden)

        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], 3)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], 3)

        edge_encs = []
        for i in range(start_idx, self.edge_type_num):
            edge_enc = self.edge_encoders[i](tmp.view(B * N * N, -1)).view(B, N, N, 1, self.nf_hidden)
            edge_encs.append(edge_enc)
        # edge_enc: B x N x N x edge_type_num x nf
        edge_enc = torch.cat(edge_encs, 3)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

        # edge_enc: B x N x N x nf
        edge_enc = edge_enc.sum(3)

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)

            edge_effects = []
            for j in range(start_idx, self.edge_type_num):
                edge_effect = self.edge_propagators[i][j](tmp.view(B * N * N, -1))
                edge_effect = edge_effect.view(B, N, N, 1, self.nf_hidden)
                edge_effects.append(edge_effect)
            # edge_effect: B x N x N x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, 3)

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

            # edge_effect: B x N x N x nf
            edge_effect = edge_effect.sum(3)

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(B, N, self.nf_hidden)

        node_effect = torch.cat([node_effect, node_enc], 2).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], 3).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred