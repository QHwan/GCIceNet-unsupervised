import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCAE(nn.Module):
    def __init__(self, n_feat, n_hid, n_lat, dropout):
        super(GCAE, self).__init__()

        n_hid1 = 64
        n_hid2 = 32
        n_hid3 = 16

        self.encoder1 = GraphConvolution(n_feat, n_hid1)
        self.encoder2 = GraphConvolution(n_hid1, n_hid2)
        self.encoder3 = GraphConvolution(n_hid2, n_hid3)
        self.encoder4 = GraphConvolution(n_hid3, n_lat)

        self.decoder1 = GraphConvolution(n_lat, n_hid3)
        self.decoder2 = GraphConvolution(n_hid3, n_hid2)
        self.decoder3 = GraphConvolution(n_hid2, n_hid1)
        self.decoder4 = GraphConvolution(n_hid1, n_feat)

        self.dropout = dropout

    def forward(self, x, adj, inv_adj):
        x = F.leaky_relu(self.encoder1(x, adj))
        x = F.leaky_relu(self.encoder2(x, adj))
        x = F.leaky_relu(self.encoder3(x, adj))
        lat = self.encoder4(x, adj)

        x = F.leaky_relu(self.decoder1(lat, adj))
        x = F.leaky_relu(self.decoder2(x, adj))
        x = F.leaky_relu(self.decoder3(x, adj))
        x = self.decoder4(x, adj)

        return lat, x


class AE(nn.Module):
    def __init__(self, n_feat, n_hid, n_lat, dropout):
        super(AE, self).__init__()

        n_hid1 = 10
        n_hid2 = 8
        n_hid3 = 6

        self.encoder1 = nn.Linear(n_feat, n_hid1)
        self.encoder2 = nn.Linear(n_hid1, n_hid2)
        self.encoder3 = nn.Linear(n_hid2, n_hid3)
        self.encoder4 = nn.Linear(n_hid3, n_lat)

        self.decoder1 = nn.Linear(n_lat, n_hid3)
        self.decoder2 = nn.Linear(n_hid3, n_hid2)
        self.decoder3 = nn.Linear(n_hid2, n_hid1)
        self.decoder4 = nn.Linear(n_hid1, n_feat)

    def forward(self, x, adj, inv_adj):
        x = F.leaky_relu(self.encoder1(x))
        x = F.leaky_relu(self.encoder2(x))
        x = F.leaky_relu(self.encoder3(x))
        lat = self.encoder4(x)

        x = F.leaky_relu(self.decoder1(lat))
        x = F.leaky_relu(self.decoder2(x))
        x = F.leaky_relu(self.decoder3(x))
        x = self.decoder4(x)
        return lat, x
