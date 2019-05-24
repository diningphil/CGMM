import torch
from torch.nn import *
import numpy as np
import matplotlib.pyplot as plt
from model_selection.EarlyStopper import GLStopper


class GeneralClassifier(Module):

    def forward(self, x):
        raise NotImplementedError('You need to implement this!')

    def train(self, tr_loader, tr_target_loader, learning_rate, l2, max_epochs,
              vl_loader=None, vl_target_loader=None, te_loader=None, te_target_loader=None,
              early_stopping=0, plot=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2)
        early_stopper = GLStopper(self) if early_stopping else None
        stop_epoch = None
        tr_loss = []
        vl_loss = []
        tr_acc = []
        vl_acc = []
        te_loss = []
        te_acc = []
        for epoch in range(1, max_epochs + 1):

            for in_data, targets in zip(tr_loader, tr_target_loader):
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # In case the dataloader took a TensorDataset in input
                if isinstance(in_data, list):
                    assert len(in_data) == 1
                    in_data = in_data[0]

                in_data = in_data.float()
                targets = targets[0]
                
                if targets.shape[0] == 1:
                    targets = targets[:, 0]
                else:
                    targets = torch.squeeze(targets)
                targets = targets.type(torch.LongTensor)

                # Forward pass to get output/logits
                outputs = self(in_data)

                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs, targets)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

            # Average of batch results
            tr_accuracy, tr_l_mean, tr_l_std = self.compute_accuracy(tr_loader, tr_target_loader)
            tr_loss.append(tr_l_mean)
            tr_acc.append(tr_accuracy)
            if vl_loader is not None:
                vl_accuracy, vl_l_mean, vl_l_std = self.compute_accuracy(vl_loader, vl_target_loader)
                vl_loss.append(vl_l_mean)
                vl_acc.append(vl_accuracy)
            if te_loader is not None:
                te_accuracy, te_l_mean, te_l_std = self.compute_accuracy(te_loader, te_target_loader)
                te_loss.append(te_l_mean)
                te_acc.append(te_accuracy)

            #'''
            if epoch % 100 == 0:
                # Print Loss
                print('Epoch: {}. TR Loss: {}. Accuracy: {}'.format(epoch, tr_l_mean, tr_accuracy))
                if vl_loader is not None:
                    print('Epoch: {}. VL Loss: {}. Accuracy: {}'.format(epoch, vl_l_mean, vl_accuracy))
                if te_loader is not None:
                    print('Epoch: {}. TE Loss: {}. Accuracy: {}'.format(epoch, te_l_mean, te_accuracy))
            #'''
            if early_stopping != 0 and epoch >= early_stopping:

                # If you want to use accuracy
                vl_acc_early_stop = 1 - (vl_accuracy/100)
                #stop = early_stopper.stop(vl_acc_early_stop, epoch)

                stop = early_stopper.stop(vl_l_mean, epoch)

                if stop:
                    print("Early stopping, stopping at epoch", epoch)
                    stop_epoch = epoch
                    break

        if plot:
            plt.plot(tr_loss, 'r')
            if vl_loader is not None:
                plt.plot(vl_loss, 'b')
            if te_loader is not None:
                plt.plot(te_loss, 'g')
            plt.figure()
            plt.plot(tr_acc, 'r')
            if vl_loader is not None:
                plt.plot(vl_acc, 'b')
            if te_loader is not None:
                plt.plot(te_acc, 'g')
        plt.show()

        if early_stopping and stop:
            self.load_state_dict(early_stopper.get_best_params())
            epoch = early_stopper.get_best_epoch()

        tr_acc, tr_l_mean, _ = self.compute_accuracy(tr_loader, tr_target_loader)
        # Print Loss
        print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, tr_l_mean, tr_acc))

        return tr_acc, stop_epoch

    def compute_accuracy(self, input_loader, target_loader):
        correct = 0
        total = 0

        # Calculate Accuracy
        # Iterate through test dataset
        with torch.no_grad():
            loss = []
            for in_data, targets in zip(input_loader, target_loader):

                # In case the dataloader took a TensorDataset in input
                if isinstance(in_data, list):
                    assert len(in_data) == 1
                    in_data = in_data[0]

                in_data = in_data.float()
                targets = targets[0]
                if len(targets.shape) > 1 and targets.shape[1] == 1:
                    targets = torch.squeeze(targets, dim=1)
                targets = targets.type(torch.LongTensor)

                # Forward pass only to get logits/output
                outputs = self(in_data)

                l = self.criterion(outputs, targets)
                loss.append(float(l))

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += targets.size(0)

                # Total correct predictions
                correct += (predicted == targets).sum()

        accuracy = 100 * correct.item() / total

        loss = np.array(loss)
        return accuracy, loss.mean(), loss.std()
