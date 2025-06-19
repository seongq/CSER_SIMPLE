import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import degree, add_self_loops

def generate_random_mask(num_nodes, num_features, mask_prob=0.5):
    """
    mask가 나옴
    :param num_nodes: node의 수
    :param num_features: node마다 feature 차원 (dimension)
    :param mask_prob: mask를 만들기 위한 parameter, mask_prob가 클 수록 더 많이 mask됨
    :return: mask가 나온다. num_nodes X num_feature 크기의 텐서가 나온다.
    """
    mask = torch.rand((num_nodes, num_features)) > mask_prob  # 生成布尔掩码
    return mask

class GraphGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add',  graph_masking=True):
        super(GraphGCN, self).__init__(aggr='add')  # "Add" aggregation.
        
        self.graph_masking = graph_masking
        
        
        self.gate = torch.nn.Linear(2*in_channels, 1)
    def forward(self, x, edge_index):
        num_nodes, dim = x.shape
        if self.graph_masking:
            mask = generate_random_mask(num_nodes, dim, mask_prob=0.5).to(x.device)
            x = x * mask

        # For original GCN, use A+I
        # if self.original_gcn:
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #     x = self.linear(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # if self.original_gcn:
        #     return norm.view(-1, 1) * x_j
        # else:
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))
        return norm.view(-1, 1) * x_j * alpha_g

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out