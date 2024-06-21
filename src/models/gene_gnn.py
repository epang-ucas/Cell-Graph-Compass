import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class GeneConv(MessagePassing):
    '''
    Graph convolution / message passing among Genes.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.    
    :param in_dims: input node embedding dimensions 
    :param out_dims: output node embedding dimensions 
    :param edge_dims: input edge embedding dimensions 
    :param n_layers: number of linear layers in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                a masked autoregressive decoder architecture
    '''
    def __init__(
        self, 
        in_dims: int, 
        out_dims: int, 
        edge_dims: int, 
        n_layers: int = 3,
        module_list = None, 
        aggr: str = "mean", 
        activations = F.relu,
        add_self_loops: bool = False,
    ):
        super(GeneConv, self).__init__(aggr=aggr)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.edge_dims = edge_dims
        self.act = activations
        self.add_self_loops = add_self_loops
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    nn.Linear(2*in_dims + edge_dims, out_dims))
                # module_list.append(self.act)
            else:
                module_list.append(
                    nn.Linear(2*in_dims + edge_dims, out_dims)
                )
                # import pdb;pdb.set_trace()
                module_list.append(self.act())
                for _ in range(n_layers - 2):
                    module_list.append(nn.Linear(out_dims, out_dims))
                    module_list.append(self.act())
                module_list.append(nn.Linear(out_dims, out_dims))
                # module_list.append(self.act)
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: torch.Tensor of shape [bsz*n_node, emb]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: torch.Tensor of shape [n_edges, edge_dims]
        '''
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value='mean', num_nodes=x.shape[-2]
            )
        # import pdb;pdb.set_trace()
        message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return message 

    def message(self, x_i, x_j, edge_attr):
        message = torch.cat([x_j, edge_attr, x_i], dim=-1)
        message = self.message_func(message)
        return message


class GeneConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer among genes.
    Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GeneConv`.
    
    :param node_dims: node embedding dimensions 
    :param edge_dims: input edge embedding dimensions 
    :param n_message: number of layers to use in message function
    :param n_feedforward: number of layers to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GeneConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    '''
    def __init__(
        self,
        node_dims: int, 
        edge_dims: int,
        n_message: int = 3, 
        n_feedforward: int = 0, 
        drop_rate = .1, 
        autoregressive = False,
        activations = nn.ReLU,
        n_edge_gvps: int = 3, 
        layernorm = True,
        add_self_loops = False,
    ):
        super(GeneConvLayer, self).__init__()
        self.n_feedforward = n_feedforward
        self.conv = GeneConv(
            node_dims,
            node_dims,
            edge_dims,
            n_layers = n_message,
            aggr = "add" if autoregressive else "mean",
            activations = activations,
            add_self_loops = add_self_loops,
        )
        if layernorm:
            self.norm = nn.ModuleList([nn.LayerNorm(node_dims) for _ in range(2)])
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(nn.Linear(node_dims, node_dims))
        elif n_feedforward != 0:
            hid_dims = 4 * node_dims
            ff_func.append(nn.Linear(node_dims, hid_dims))
            ff_func.append(activations())
            for _ in range(n_feedforward-2):
                ff_func.append(nn.Linear(hid_dims, hid_dims))
                ff_func.append(activations())
            ff_func.append(nn.Linear(hid_dims, node_dims))
        self.ff_func = nn.Sequential(*ff_func)

        module_list = []
        if n_edge_gvps == 1:
            module_list.append(nn.Linear(2*node_dims + edge_dims, edge_dims))
        elif n_edge_gvps > 1:
            module_list.append(nn.Linear(2*node_dims + edge_dims, edge_dims))
            module_list.append(activations())
            for _ in range(n_edge_gvps - 2):
                module_list.append(nn.Linear(edge_dims, edge_dims))
                module_list.append(activations())
            module_list.append(nn.Linear(edge_dims, edge_dims))
        self.edge_message_func = nn.Sequential(*module_list)
        if layernorm:
            self.edge_norm = nn.LayerNorm(edge_dims)
        else:
            self.edge_norm = nn.Identity()
        self.edge_dropout = nn.Dropout(drop_rate)
    
    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: node embedding of shape [bsz*n_node, node_dims]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: edge embedding of shape [n_edges, edge_dims] 
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as srcqq node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        # import pdb;pdb.set_trace()
        if self.edge_message_func:
            src, dst = edge_index
            if autoregressive_x is None:
                x_src = x[src]
            else: 
                mask = (src < dst).unsqueeze(-1)
                x_src = (
                    torch.where(mask, x[0][src], autoregressive_x[0][src]),
                    torch.where(mask.unsqueeze(-1), x[1][src],
                        autoregressive_x[1][src])
                )
            x_dst = x[dst]
            # import pdb;pdb.set_trace()
            x_edge = torch.cat([x_src, edge_attr, x_dst], dim=-1)
 
            edge_attr_dh = self.edge_message_func(x_edge)
            edge_attr = self.edge_norm(edge_attr + self.edge_dropout(edge_attr_dh))
        
        if autoregressive_x is not None:
            src, dst = edge_index
            # mask = src < dst
            # edge_index_forward = edge_index[:, mask]
            # edge_index_backward = edge_index[:, ~mask]
            # edge_attr_forward = tuple_index(edge_attr, mask)
            # edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            # dh = tuple_sum(
            #     self.conv(x, edge_index_forward, edge_attr_forward),
            #     self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            # )
            
            # count = scatter_add(torch.ones_like(dst), dst,
            #             dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            # dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
            
        
        if node_mask is not None:
            x_ = x
            x, dh = x[node_mask], dh[node_mask]
            
        x = self.norm[0](x + self.dropout[0](dh))
        if self.n_feedforward:
            dh = self.ff_func(x)
            x = self.norm[1](x + self.dropout[1](dh))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x, edge_attr