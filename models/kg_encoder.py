import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class KGEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, node_types, metadata):
        """
        Heterogeneous Graph Neural Network for Knowledge Graph Embeddings.
        
        Args:
            hidden_channels (int): Hidden dimensions.
            out_channels (int): Output dimensions (size of embeddings z_KG).
            num_layers (int): Number of Message Passing layers.
            node_types (list): List of node type strings.
            metadata (tuple): (node_types, edge_types) from HeteroData object.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                # edge_type is a tuple like ('mab', 'targets', 'target')
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_channels)
            
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        self.out_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.out_lin_dict[node_type] = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Linear embedding of initial node features
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = F.relu(self.lin_dict[node_type](x))

        # 2. Message Passing layers
        for conv in self.convs:
            new_out_dict = conv(out_dict, edge_index_dict)
            for key in out_dict.keys():
                if key not in new_out_dict:
                    new_out_dict[key] = out_dict[key]
            out_dict = {key: F.relu(x) for key, x in new_out_dict.items()}

        # 3. Output projections
        final_out_dict = {}
        for node_type, x in out_dict.items():
            final_out_dict[node_type] = self.out_lin_dict[node_type](x)

        return final_out_dict
