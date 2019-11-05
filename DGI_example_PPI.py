from __future__ import print_function
import os.path as osp
import random
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import SAGEConv, DeepGraphInfomax

import json
import numpy as np

from networkx.readwrite import json_graph
from argparse import ArgumentParser

#dataset = 'Cora'
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
#dataset = Planetoid(path, dataset)

class Encoder(nn.Module):
    def __init__(self, dim_features, dim_embedding):
        super(Encoder, self).__init__()
        #self.conv = SAGEConv(dim_features, dim_embedding)
        #self.prelu = nn.PReLU(dim_embedding)

        self.prelu = nn.PReLU()  # like in Repository
        self.gcn1 = SAGEConv(dim_features, dim_embedding, normalize=True)
        self.gcn2 = SAGEConv(dim_embedding, dim_embedding, normalize=True)
        self.gcn3 = SAGEConv(dim_embedding, dim_embedding, normalize=True)
        self.Wskip = Linear(dim_features, dim_embedding)
        
    def forward(self, x, edge_index):

        # Apply Dropout to the input features as in the paper
        x = F.dropout(x, p=0., training=self.training)

        H1skip = self.Wskip(x)
        h_1_1 = self.prelu(self.gcn1(x, edge_index))
        h_1_2 = self.prelu(self.gcn2(h_1_1 + H1skip, edge_index))
        h_1_3 = self.prelu(self.gcn3(h_1_2 + H1skip, edge_index))

        x = self.prelu(h_1_3)
        return x


dataset = PPI('train')

test_set = PPI('test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# todo training on the first graph only!
#data = dataset[0].to(device)

# Used for other datasets
#def corruption(x, edge_index):
#    return x[torch.randperm(x.size(0))], edge_index

# Used for PPI
def corruption(x, edge_index):
    # Ignore parameters
    rand_idx = random.randint(0, len(dataset)-1)
    return dataset[rand_idx].x.to(device), dataset[rand_idx].edge_index.to(device)

model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fun = nn.BCELoss()

def train():
    model.train()
   
    # This is the only way in which the algorithm converges
    rand_idx = 0#random.randint(0, len(dataset)-1)

    # Generate random permutation
    perm = [i for i in range(len(dataset))]
    random.shuffle(perm)
    #print(perm)

    all_loss = 0.

    for idx in perm:  # Batch size 1 for now
        
        optimizer.zero_grad()
        
        # COMPARE THE FIRST WITH ALL THE REST!

        data = dataset[rand_idx].to(device)

        pos_z, neg_z, summary = model(data.x, data.edge_index)
        
        #loss = model.loss(pos_z, neg_z, summary)

        pos, neg = model.discriminate(pos_z, summary, sigmoid=True), model.discriminate(neg_z,summary, sigmoid=True)
        loss = loss_fun(torch.cat((pos, neg), dim=0), torch.cat((torch.ones(pos_z.shape[0]), torch.zeros(neg_z.shape[0])), dim=0).to(device))

        loss.backward()
        optimizer.step()

        all_loss += loss.item()

    return all_loss/len(perm)


def test(dataset_eval):
    model.eval()

    z_all = None
    y_all = None

    for idx in range(len(dataset_eval)):

        data = dataset[idx].to(device)

        z, _, _ = model(data.x, data.edge_index)

        if z_all is None:
            z_all = z
        else:
            z_all = torch.cat((z_all, z), dim=0)

        if y_all is None:
            y_all = data.y
        else:
            y_all = torch.cat((y_all, data.y), dim=0)

    return z_all, y_all


# Train
for epoch in range(1, 20):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

# Compute embeddings
train_emb, train_y = test(dataset)
test_emb, test_y = test(test_set)

train_emb = train_emb.detach().cpu().numpy()
train_y = train_y.detach().cpu().numpy()
test_emb = test_emb.detach().cpu().numpy()
test_y = test_y.detach().cpu().numpy()

# Standardize embeddings
mean = train_emb.mean(axis=0)
std = train_emb.std(axis=0)

train_emb = (train_emb - mean) / std
test_emb = (test_emb - mean) / std


''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
'''

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    #np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    log.fit(train_embeds, train_labels)

    y_pred_test = log.predict(test_embeds)
    print(y_pred_test.shape)
    print("F1 score", f1_score(test_labels, y_pred_test, average="micro"))

    print("Random baseline F1 score", f1_score(test_labels, dummy.predict(test_embeds), average="micro"))

run_regression(train_emb, train_y, test_emb, test_y)

