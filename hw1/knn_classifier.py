import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import hw1.dataloaders as hw1dataloader
import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        n_classes = 0
        x_train, y_train = torch.Tensor(), torch.Tensor()
        for x,y in dl_train:
            x_train = torch.cat((x_train, x),0)
            y_train = torch.cat((y_train, y),0)
            n_classes = max(y.max(), n_classes)
        # ========================
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            topk_ind = torch.topk(dist_matrix[:, i], self.k, dim=0, largest=False).indices
            topk = self.y_train[topk_ind]
            y_pred[i] = torch.mode(topk, 0).values
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # dists = torch.sqrt(torch.sum((x1[:,None,...]-x2[None,...])**2, dim=-1))
    x1_norm = (x1**2).sum(1).view(-1, 1)
    x2_norm = (x2**2).sum(1).view(1, -1)
    dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, torch.transpose(x2, 0, 1))
    return torch.sqrt(dist)


    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    # print(len(torch.where(y_pred==y)[0]))
    accuracy = len(torch.where(y_pred==y)[0])/len(y)
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """
# TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.
        
    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)
        fold_acc = []
        for k in range(num_folds):
            val_len = int(len(ds_train)*0.2)
            dl_train, dl_val = hw1dataloader.create_train_validation_loaders(ds_train, 0.2, batch_size=val_len)
            model.train(dl_train)
            x_val, y_val = next(iter(dl_val))
            y_pred = model.predict(x_val)
            fold_acc.append(accuracy(y_val, y_pred))
        accuracies.append(fold_acc)
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
