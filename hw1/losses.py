import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N,C =x_scores.shape
        # self.x = x
        y_score =  x_scores[torch.arange(N),y]
        mask = torch.ones_like(x_scores).scatter_(1, y.unsqueeze(1), 0.).bool()
        # print(mask.shape)
        M = x_scores - y_score[:,None] + self.delta
        res = M[mask].view(N, C-1)
        # res = torch.maximum(res, torch.zeros_like(res)).sum(dim=1)
        res = torch.clamp(res, min=0).sum(dim=1)
        loss = torch.sum(res)/N
        self.grad_ctx['x'] = x
        self.grad_ctx['M'] = M
        self.grad_ctx['mask'] = mask
        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        g_dict = self.grad_ctx
        M, mask = g_dict['M'], g_dict['mask']
        g = torch.zeros_like(M)
        g1 = M[mask].view(g.shape[0], g.shape[1]-1)
        g1[g1 > 0] = 1
        # print(g.shape, g1.shape, g_dict['mask'].shape)
        g[mask] = g1.view(-1)
        # g[g_dict['mask']][g > 0] = 1
        # g2 = g[torch.logical_not(mask)]
        # print(g2.shape)
        g[torch.logical_not(mask)] = -g1.sum(dim=1)
        # print(g.shape, g_dict['x'].shape)
        grad = torch.mean(g[...,None]*g_dict['x'][:,None,:], dim=0)
        # print(grad.shape)
        # g1 = self.M[self.mask]
        # g1[g1 > 0] = 1
        # g1_plus_ind = torch.where(g1 > 0)
        # g1[g1 <= 0] = 0
        # g2 = self.M[torch.logical_not(self.mask)]
        # g2[g2 > 0] = -1*g1.sum(dim=1)
        
        # ========================

        return grad.T
