from models.graph_classifiers.DGIEncoder import DGIEncoder
from experiments.Experiment import Experiment
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import DeepGraphInfomax
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

# One epoch of training
def train(model, dataset, device, loss_fun, optimizer, training_strategy):
    model.train()
   
    # Comparing one graph against the others is the only way in which DGI converges, ow, as in the paper, it performs similarly to random.
    # We believe the authors were not able to make DGI converge on PPI, and they simply computed an incorrect micro average F1 score.
    if training_strategy == 'one_vs_all':
        rand_idx = 0
    elif training_strategy == 'all_vs_all':
        rand_idx = random.randint(0, len(dataset)-1)
    
    # Generate random permutation
    perm = [i for i in range(len(dataset))]
    random.shuffle(perm)

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


def eval(model, dataset_eval, device):
    model.eval()

    z_all = None
    y_all = None

    for idx in range(len(dataset_eval)):

        data = dataset_eval[idx].to(device)

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



''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
'''

def run_regression(train_embeds, train_labels, validation_embeds=None, validation_labels=None, test_embeds=None, test_labels=None):
    log = MultiOutputClassifier(SGDClassifier(loss="log", early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, tol=1e-3),
                                n_jobs=10) # Logistic regression

    if validation_embeds is not None:
        log.fit(np.concatenate((train_embeds, validation_embeds), axis=0),
                np.concatenate((train_labels, validation_labels), axis=0))
    else:
        log.fit(train_embeds, train_labels)

    y_pred_train= log.predict(train_embeds)
    train_score = f1_score(train_labels, y_pred_train, average="micro")


    if validation_embeds is not None:
        y_pred_validation = log.predict(validation_embeds)
        validation_score = f1_score(validation_labels, y_pred_validation, average="micro")
    else:
        validation_score = 0.

    if test_embeds is not None:
        y_pred_test = log.predict(test_embeds)
        test_score = f1_score(test_labels, y_pred_test, average="micro")
    else:
        test_score = 0.

    return train_score, validation_score, test_score


class DGIPPIExperiment(Experiment):

    def __init__(self, model_config, exp_path):
        super(DGIPPIExperiment, self).__init__(model_config, exp_path)

    def run_valid(self, dataset_getter, logger, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        dataset_class = self.model_config.dataset  # dataset_class()
        dataset = dataset_class()
        
        device = self.model_config.device
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'], shuffle=shuffle)

        def corruption(x, edge_index, dataset):
            # Ignore parameters, we do not want to shuffle the graph features as in single graph tasks
            rand_idx = random.randint(0, len(dataset)-1)
            return dataset[rand_idx].x.to(device), dataset[rand_idx].edge_index.to(device)

        model = DeepGraphInfomax(
            hidden_channels=self.model_config['dim_embedding'], encoder=DGIEncoder(dataset._dim_features, self.model_config['dim_embedding'], self.model_config),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index: corruption(x, edge_index, train_loader.dataset)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        loss_fun = nn.BCELoss()

        # Train
        for epoch in range(1, self.model_config['DGI_epochs']+1):
            loss = train(model, train_loader.dataset, device, loss_fun, optimizer, self.model_config['training_strategy'])
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

        # Compute embeddings
        train_emb, train_y = eval(model, train_loader.dataset, device)
        valid_emb, valid_y = eval(model, val_loader.dataset, device)

        train_emb = train_emb.detach().cpu().numpy()
        train_y = train_y.detach().cpu().numpy()
        valid_emb = valid_emb.detach().cpu().numpy()
        valid_y = valid_y.detach().cpu().numpy()

        # Standardize embeddings
        mean = train_emb.mean(axis=0)
        std = train_emb.std(axis=0)

        train_emb = (train_emb - mean) / std
        valid_emb = (valid_emb - mean) / std

        train_score, valid_score, _ = run_regression(train_emb, train_y, valid_emb, valid_y, test_embeds=None, test_labels=None)

        print(f'MICRO F1 TRAIN SCORE: {train_score} \t VALIDATION SCORE: {valid_score}')

        return train_score, valid_score

    def run_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """

        dataset_class = self.model_config.dataset  # dataset_class()
        dataset = dataset_class()
        
        device = self.model_config.device
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        def corruption(x, edge_index, dataset):
            # Ignore parameters, we do not want to shuffle the graph features as in single graph tasks
            rand_idx = random.randint(0, len(dataset)-1)
            return dataset[rand_idx].x.to(device), dataset[rand_idx].edge_index.to(device)

        model = DeepGraphInfomax(
            hidden_channels=self.model_config['dim_embedding'], encoder=DGIEncoder(dataset._dim_features, self.model_config['dim_embedding'], self.model_config),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index: corruption(x, edge_index, train_loader.dataset)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        loss_fun = nn.BCELoss()

        # Train
        for epoch in range(1, self.model_config['DGI_epochs']+1):
            loss = train(model, train_loader.dataset, device, loss_fun, optimizer, self.model_config['training_strategy'])
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

        # Compute embeddings
        train_emb, train_y = eval(model, train_loader.dataset, device)
        valid_emb, valid_y = eval(model, val_loader.dataset, device)
        test_emb, test_y = eval(model, test_loader.dataset, device)

        train_emb = train_emb.detach().cpu().numpy()
        train_y = train_y.detach().cpu().numpy()
        valid_emb = valid_emb.detach().cpu().numpy()
        valid_y = valid_y.detach().cpu().numpy()
        test_emb = test_emb.detach().cpu().numpy()
        test_y = test_y.detach().cpu().numpy()

        # Standardize embeddings
        mean = train_emb.mean(axis=0)
        std = train_emb.std(axis=0)

        train_emb = (train_emb - mean) / std
        valid_emb = (valid_emb - mean) / std
        test_emb = (test_emb - mean) / std

        train_score, validation_score, test_score = run_regression(train_emb, train_y, valid_emb, valid_y, test_emb, test_y)

        print(f'MICRO F1 TRAIN SCORE: {train_score} \t VALID SCORE: {validation_score} \t TEST SCORE: {test_score}')

        return train_score, test_score




        return train_acc, test_acc
