import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.nn.Parameter(torch.Tensor(self.n_features, self.n_classes))
        torch.nn.init.normal_(self.weights, std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=-1)
        # print(class_scores.shape)
        # print(y_pred.shape, y_pred)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = (y==y_pred).sum().item()/len(y)
        # ========================

        return acc * 100


    def train(
            self,
            dl_train: DataLoader,
            dl_valid: DataLoader,
            loss_fn: ClassifierLoss,
            learn_rate=0.1,
            weight_decay=0.001,
            max_epochs=100,
        ):

            Result = namedtuple("Result", "accuracy loss")
            train_res = Result(accuracy=[], loss=[])
            valid_res = Result(accuracy=[], loss=[])

            print("Training", end="")
            for epoch_idx in range(max_epochs):
                total_correct = 0
                average_loss = 0

                train_acc = 0
                train_loss = 0
                for x_train,y_train in dl_train:
                    y_predict, class_scores = self.predict(x_train)
                    train_loss += loss_fn.loss(x_train, y_train, class_scores, y_predict) + 0.5*weight_decay*torch.norm(self.weights)
                    self.weights = self.weights - learn_rate *(loss_fn.grad()+ weight_decay*self.weights)
                    train_acc += self.evaluate_accuracy(y_train, y_predict)
                train_res.accuracy.append(train_acc/len(dl_train))
                train_res.loss.append(train_loss.detach()/len(dl_train))
                # print('/n', train_loss, train_acc/len(dl_train))

                val_acc = 0
                val_loss = 0
                for x_val, y_val in dl_valid:
                    y_predict, class_scores = self.predict(x_val)
                    val_loss += loss_fn.loss(x_val, y_val, class_scores, y_predict) + 0.5*weight_decay*torch.norm(self.weights)
                    val_acc += self.evaluate_accuracy(y_val, y_predict)
                valid_res.accuracy.append(val_acc/len(dl_valid))
                valid_res.loss.append(val_loss.detach()/len(dl_valid))


                # ========================
                print(".", end="")

            print("")
            return train_res, valid_res
    
    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        # if has_bias:
        n_features = self.n_features if not has_bias else self.n_features - 1
        w_images = self.weights[:n_features, :].t().view(self.n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.001, learn_rate=0.02, weight_decay=0.1)
    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    # ========================

    return hp
