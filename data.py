import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
import torch
import pickle


def load_data(path="data/", name_list=['w', '1h', '1c', '2', '3', '6', '7', 'T', 'sI'], n_conf=10):
    print('Loading ice dataset...')

    adj_list = []
    inv_adj_list = []
    feature_list = []
    label_list = []

    idx = np.linspace(0, n_conf-1, n_conf)

    n_class = len(name_list)

    for i, name in enumerate(name_list):
        data_fname = path + "data_{}_10frame.npz".format(name)
        adj_fname = path + "a_hat_{}_10frame.pickle".format(name)
        inv_adj_fname = path + "inv_a_hat_{}_10frame.pickle".format(name)

        data_file = np.load(data_fname)

        feat_list = data_file['feature']
        n_fea = np.shape(feat_list)[2]

        with open(adj_fname, 'rb') as f:
            adj_sp_list = pickle.load(f)
        with open(inv_adj_fname, 'rb') as f:
            inv_adj_sp_list = pickle.load(f)

        n_node = np.shape(feat_list)[1]

        feat_list = feat_list[:n_conf]
        adj_sp_list = adj_sp_list[:n_conf]
        inv_adj_sp_list = inv_adj_sp_list[:n_conf]

        for j, (feat, adj_sp, inv_adj_sp) in enumerate(zip(feat_list, adj_sp_list, inv_adj_sp_list)):

            feature = feat

            ### delete ###
            feature[:6] = 0.5
            feature[:4] = 0.5

            label = np.full(n_node, i)

            feature = torch.FloatTensor(feature)
            label = torch.LongTensor(label)
            adj = adj_sp.todense()
            inv_adj = inv_adj_sp.todense()
            adj = torch.FloatTensor(np.array(adj))
            inv_adj = torch.FloatTensor(np.array(inv_adj))

            if j in idx:
                adj_list.append(adj)
                inv_adj_list.append(inv_adj)
                feature_list.append(feature)
                label_list.append(label)


    # MinMaxScaling
    for i in range(n_fea):
        feature_vec = []

        for j, feature in enumerate(feature_list):
            for fea in feature[:,i]:
                feature_vec.append(fea)

        min_f = np.min(feature_vec)
        max_f = np.max(feature_vec)
        df = max_f - min_f

        for j, feature in enumerate(feature_list):
            feature[:,i] = (feature[:,i]-min_f)/df


    return adj_list, inv_adj_list, feature_list, label_list, n_fea, n_class



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def output_to_label(output):
    preds = output.max(1)[1]
    preds = preds.detach().numpy()
    return preds

def prepare_svm_data(features_data, labels_data):
    x = []
    y = []
    for feature in features_data:
        feature = feature.numpy()
        for f in feature:
            x.append(f)
    for label in labels_data:
        label = label.numpy()
        for l in label:
            y.append(l)
    return x, y

