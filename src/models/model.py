import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d, Linear, Dropout
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap

torch.manual_seed(42)  
'''
    The Hitchhiker's Guide to the Galaxy
        42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything." 😁
'''

class GNN_GAT(torch.nn.Module):
    def __init__(self, feature_size, parameters):
        super(GNN_GAT, self).__init__()
        embed_size = parameters["model_embed_size"]
        num_heads = parameters["model_num_heads"]
        dropout_gat = parameters["model_dropout_gat"]
        self.num_layers = parameters["model_num_layers"]
        edge_dim = parameters["model_edge_dim"]
        topKratio = parameters["model_topKratio"]
        dense_neurons = parameters["model_dense_neurons"]
        dense_dropout_rate = parameters["model_dense_dropout_rate"]

        self.conv_layers = ModuleList([])
        self.linear_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.pooling_layers = ModuleList([])

        self.conv1 = GATConv(in_channels=feature_size,
                             out_channels=embed_size,
                             heads=num_heads,
                             dropout=dropout_gat, edge_dim=edge_dim)
        
        self.linear1 = Linear(in_features=embed_size * num_heads, out_features=embed_size)
        self.bn1 = BatchNorm1d(num_features=embed_size)
        self.TKpool1 = TopKPooling(in_channels=embed_size, ratio=topKratio)


        for i in range(self.num_layers-1):
            self.conv_layers.append(GATConv(in_channels=embed_size,
                             out_channels=embed_size,
                             heads=num_heads,
                             dropout=dropout_gat, edge_dim=edge_dim))
            
            self.linear_layers.append(Linear(in_features=embed_size * num_heads, out_features=embed_size))

            self.bn_layers.append(BatchNorm1d(num_features=embed_size))

            self.pooling_layers.append(TopKPooling(in_channels=embed_size, ratio=topKratio))


        self.dense1 = Linear(in_features=embed_size * 2, out_features=dense_neurons)
        self.drop1 = Dropout(p=dense_dropout_rate)
        self.dense2 = Linear(in_features=dense_neurons, out_features=int(dense_neurons/2))
        self.drop2 = Dropout(p=dense_dropout_rate)
        self.dense3 = Linear(in_features=int(dense_neurons/2), out_features=1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        gnn_representation = []

        x = self.bn1(F.relu(self.linear1( self.conv1(x, edge_index, edge_attr))))
        x, edge_index, edge_attr, batch_index, _, _ = self.TKpool1(x, edge_index, edge_attr, batch_index)

        gnn_representation.append(torch.cat((gap(x, batch_index), gmp(x, batch_index)), dim=1))

        for i in range(self.num_layers - 1):
            x = self.bn_layers[i](F.relu(self.linear_layers[i]( self.conv_layers[i](x, edge_index, edge_attr))))
            x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[i](x, edge_index, edge_attr, batch_index)

            gnn_representation.append(torch.cat((gap(x, batch_index), gmp(x, batch_index)), dim=1))
        
        x = sum(gnn_representation)

        x = self.dense3(self.drop2(self.dense2(self.drop1(self.dense1(x)))))

        return x.squeeze()
       




        
