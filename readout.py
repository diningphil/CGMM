import torch
from pydgn.model.interface import ReadoutInterface


class CGMMGraphReadout(ReadoutInterface):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

        embeddings_node_features = dim_node_features
        hidden_units = config['hidden_units']

        self.fc_global = torch.nn.Linear(embeddings_node_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        out = self.out(torch.relu(self.fc_global(data.x.float())))
        return out, data.x
