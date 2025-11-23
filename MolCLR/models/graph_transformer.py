import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool, global_max_pool


num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3

class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_layer=5,
        emb_dim=300,
        feat_dim=256,
        heads=4,
        drop_ratio=0.1,
        pool='mean',
        edge_emb_dim=32,   
        use_3D = False
    ):
        super().__init__()
        assert num_layer >= 2, "Number of layers must be >= 2"
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.heads = heads
        self.edge_emb_dim = edge_emb_dim
        self.use_3D = use_3D


        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)


        self.edge_embedding_type = nn.Embedding(num_bond_type, edge_emb_dim)
        self.edge_embedding_dir  = nn.Embedding(num_bond_direction, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding_type.weight)
        nn.init.xavier_uniform_(self.edge_embedding_dir .weight)

       
        self.edge_proj = nn.Linear(edge_emb_dim, edge_emb_dim)
        
        if use_3D:
            self.pos_emb = nn.Linear(3, emb_dim)


        self.layers = nn.ModuleList()
        self.norms1 = nn.ModuleList() 
        self.norms2 = nn.ModuleList()  
        self.ffns   = nn.ModuleList()  

        for _ in range(num_layer):
            self.layers.append(
                TransformerConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim // heads,
                    heads=heads,
                    dropout=drop_ratio,
                    edge_dim=edge_emb_dim,    
                    beta=False           
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

   
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError("pool must be one of ['mean', 'add', 'max']")


        self.feat_lin = nn.Linear(emb_dim, feat_dim)
        self.out_lin = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim // 2),
        )

    def _edge_encode(self, edge_attr):
        """
        edge_attr: [E, 2] with columns [bond_type, bond_dir]
        returns edge_feat: [E, edge_emb_dim]
        """
        e = self.edge_embedding_type(edge_attr[:, 0].long()) + \
            self.edge_embedding_dir (edge_attr[:, 1].long())
        return self.edge_proj(e)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if self.use_3D:
            pos= data.pos
            pe = self.pos_emb(pos)
        else:
            pe = 0
        


        h = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long())
        h = h + pe


        edge_feat = self._edge_encode(edge_attr)


        for conv, ln1, ln2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):

            h_res = h
            h = conv(h, edge_index, edge_attr=edge_feat)  # [N, emb_dim]
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            h = ln1(h_res + h)

            
            h_res2 = h
            h = ffn(h)
            h = ln2(h_res2 + h)  

        g = self.pool(h, batch)      # [B, emb_dim]
        g = self.feat_lin(g)         # [B, feat_dim]
        out = self.out_lin(g)        # [B, feat_dim//2]
        return g, out


# model = GraphTransformer(num_layer=5, emb_dim=300, feat_dim=256, heads=4, drop_ratio=0.1, pool='mean', edge_emb_dim=32)
