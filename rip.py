from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA

from models import AE, GCAE
from data import load_data, accuracy, output_to_label, prepare_svm_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--latent', type=int, default=2,
                    help='Number of latent units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_conf', type=int, default=10,
                    help='Number of snapshots used.')
parser.add_argument('--model', type=str, default='GCAE',
                    help='Network Model: PCA, AE, GCAE')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')
parser.add_argument('--o', type=str, default=None,
                    help='Save file')

args = parser.parse_args()


np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))





def test():
    if args.model == 'PCA':
        pca = PCA(n_components=10)
        x, y = prepare_svm_data(features, labels)
        x_pca = pca.fit_transform(x)

        return x_pca, y

    encoded_list = []
    decoded_list = []
    label_list = []

    running_loss = 0
    
    for i in range(len(adj)):
        model.eval()
        encoded, decoded = model(features[i], adj[i], inv_adj[i])
        loss = criterion(decoded, features[i])
        running_loss += loss

        encoded = encoded.detach().numpy()
        decoded = decoded.detach().numpy()
        label = labels[i].detach().numpy()
        feature = features[i].detach().numpy()

        for j in range(len(encoded)):
            encoded_list.append(encoded[j])
            label_list.append(label[j])
        if i == 0:
            decoded_list.append(decoded[0])
            decoded_list.append(feature[0])
            decoded_list = np.transpose(decoded_list)

    print("Test set results:",
          "loss= {:.8f}".format(running_loss/len(adj)))

    return encoded_list, label_list, running_loss.detach().numpy()/len(adj)

    

perturb_list = [None, 0, 1, 2, 3, 4 ,5, 6, 7, 8, 9, 10, 11]
loss_list = []

# Load data
for p in perturb_list:
    adj, inv_adj, features, labels, n_feat, n_class = load_data(n_conf=args.n_conf, perturb=p)

    # Model and optimizer
    if args.model == 'AE' or args.model == "PCA":
        model = AE(n_feat=n_feat,
                   n_hid=args.hidden,
                   n_lat=args.latent,
                   dropout=args.dropout)
    elif args.model == 'GCAE':
        model = GCAE(n_feat=n_feat,
                    n_hid=args.hidden,
                    n_lat=args.latent,
                    dropout=args.dropout)
    else:
        raise ValueError("You choose wrong network model")

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Test
    encoded_list, label_list, loss = test()

    if p is None:
        ref_loss = loss
    else:
        loss_list.append(loss)

loss_list = np.array(loss_list)
loss_list -= ref_loss
sum_loss = np.sum(loss_list)
loss_list /= sum_loss

print(loss_list)
print(np.sum(loss_list))

if args.o is not None:
    np.savez(args.o, rip=loss_list)


