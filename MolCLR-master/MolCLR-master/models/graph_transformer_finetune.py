import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119   # including mask token
num_chirality_tag = 3
num_bond_type = 5     # including aromatic and self-loop
num_bond_direction = 3


class GraphTransformer(nn.Module):
    def __init__(
        self,
        task='classification',
        num_layer=5,
        emb_dim=300,
        feat_dim=256,
        heads=4,
        drop_ratio=0.1,
        pool='mean',
        edge_emb_dim=32,
    ):
        super(GraphTransformer, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.heads = heads
        self.task = task
        self.edge_emb_dim = edge_emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # --- Atom embeddings ---
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        # --- Edge embeddings ---
        self.edge_embedding1 = nn.Embedding(num_bond_type, edge_emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)
        self.edge_proj = nn.Linear(edge_emb_dim, edge_emb_dim)

        # --- Transformer layers ---
        self.layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(num_layer):
            self.layers.append(
                TransformerConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim // heads,
                    heads=heads,
                    dropout=drop_ratio,
                    edge_dim=edge_emb_dim,
                )
            )
            self.norms1.append(nn.LayerNorm(emb_dim))
            self.norms2.append(nn.LayerNorm(emb_dim))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(emb_dim, 4 * emb_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop_ratio),
                    nn.Linear(4 * emb_dim, emb_dim),
                    nn.Dropout(drop_ratio),
                )
            )

        # --- Pooling ---
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("Not defined pooling!")

        # --- Projection + prediction head ---
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Softplus(),
                nn.Linear(self.feat_dim // 2, 2),
            )
        elif self.task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Softplus(),
                nn.Linear(self.feat_dim // 2, 1),
            )
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def _edge_encode(self, edge_attr):
        e = self.edge_embedding1(edge_attr[:, 0].long()) + \
            self.edge_embedding2(edge_attr[:, 1].long())
        return self.edge_proj(e)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node init
        h = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long())
        edge_feat = self._edge_encode(edge_attr)

        # Transformer layers
        for conv, norm1, norm2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_feat)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            h = norm1(h_res + h)

            h_res2 = h
            h = ffn(h)
            h = norm2(h_res2 + h)

        # Graph-level pooling
        h = self.pool(h, batch)
        h = self.feat_lin(h)

        return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)


if __name__ == "__main__":
    model = GraphTransformer(task='classification')
    print(model)
