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
from data import load_data, accuracy, prepare_svm_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Initial learning rate.')
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
parser.add_argument('--log', type=str, default=None,
                    help='Log file containing training process')
parser.add_argument('--o', type=str, default=None,
                    help='Save model')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='continue training')
parser.add_argument('--encoded', type=str, default=None,
                    help='Encoded file')

args = parser.parse_args()

np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))


# Load data
adj, inv_adj, features, labels, n_feat, n_class = load_data(n_conf=args.n_conf)

# Model and optimizer
if args.model == 'AE' or args.model == 'PCA':
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

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

n = len(adj)

def train(epoch):
    t = time.time()

    running_loss = 0

    for i in range(n):
        model.train()
        optimizer.zero_grad()
        _, decoded = model(features[i], adj[i], inv_adj[i])
        loss = criterion(decoded, features[i])
        loss.backward()
        optimizer.step()
        running_loss += loss.data

    print('Epoch: {:04d}'.format(epoch+1),
          'loss: {:.8f}'.format(running_loss/n),
          'time: {:.4f}s'.format(time.time() - t))

    return [epoch+1, running_loss/n]

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
    
    for i in range(n):
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
          "loss= {:.4f}".format(running_loss/n))

    return encoded_list, label_list


# Train model
if args.model != 'PCA':
    log = []
    t_total = time.time()
    for epoch in range(args.epochs):
        log_epoch = train(epoch)
        log.append(log_epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Test
encoded_list, label_list = test()

if args.encoded is not None:
    np.savez(args.encoded, encoded=encoded_list, label=label_list)


# Save File
if args.log is not None:
    np.savetxt(args.log, log)

if args.o is not None:
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, args.o)
