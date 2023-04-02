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
        acc = len(torch.where(y_pred==y)[0])/len(y)
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
         # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # ========================

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        self.weights.requires_grad = True
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0
            average_loss_torch = 0
            for x,y in dl_train:
                y_pred, x_scores = self.predict(x)
                loss = loss_fn(x, y, x_scores, y_pred).item()
                average_loss += loss
                total_correct += self.evaluate_accuracy(y,y_pred)
                grad = loss_fn.grad() + weight_decay*self.weights
                self.weights = self.weights - learn_rate*grad
            # print(total_correct, (average_loss / len(dl_train)).item())
            # print(total_correct/len(dl_train))
            train_res.accuracy.append(total_correct / len(dl_train))
            train_res.loss.append(average_loss / len(dl_train))
            total_correct = 0
            average_loss = 0
            with torch.no_grad():
                for x,y in dl_valid:
                    y_pred, x_scores = self.predict(x)
                    loss = loss_fn(x, y, x_scores, y_pred).item()
                    average_loss += loss
                    total_correct += self.evaluate_accuracy(y,y_pred)
                valid_res.accuracy.append(total_correct / len(dl_valid))
                valid_res.loss.append((average_loss / len(dl_valid)))
            print(".", end="")
        print("")
        self.weights = self.weights.detach()
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
    hp = dict(weight_std=0.005, learn_rate=0.0017, weight_decay=1)
    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    # ========================

    return hp
