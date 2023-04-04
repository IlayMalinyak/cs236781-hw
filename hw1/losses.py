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
        
#         help0 = x_scores.gather(1, y.view(-1, 1))
#         help1 = help0.expand(-1, x_scores.size(1))
#         margins = x_scores - help1 + self.delta
#         margins[margins<0] = 0
#         mask = torch.ones_like(margins)
#         mask.scatter_(1, y.view(-1, 1), 0)  # zero out the correct class
#         losses = mask * margins
#         losses = losses.sum(dim=1)
#         loss2 = losses.mean()
        
        N,C =x_scores.shape    
        y_score =  x_scores[torch.arange(N),y]
        mask = torch.ones_like(x_scores).scatter_(1, y.unsqueeze(1), 0.).bool()
        M = x_scores - y_score[:,None] + self.delta
        res = M[mask].view(N, C-1)
        res = torch.clamp(res, min=0).sum(dim=1)
        loss = torch.sum(res)/N
        # print(loss, loss2)
        self.grad_ctx['x'] = x
        self.grad_ctx['M'] = M
        self.grad_ctx['mask'] = mask
        self.grad_ctx['x_scores'] = x_scores
        self.grad_ctx['y']=y
        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # # TODO:
        # #  Implement SVM loss gradient calculation
        # #  Same notes as above. Hint: Use the matrix M from above, based on
        # #  it create a matrix G such that X^T * G is the gradient.
        # margins = self.grad_ctx["M"]
        # mask = self.grad_ctx["mask"]
        # x = self.grad_ctx["x"]
        # x_scores = self.grad_ctx["x_scores"]
        # y = self.grad_ctx["y"]
        # grad2 = torch.zeros_like(x_scores)
        # margin_mask = (margins > 0).float()
        # margin_mask[mask == 0] = 0
        # margin_mask[torch.arange(grad2.shape[0]), y] = -margin_mask.sum(dim=1)
        # grad2 = (x.t() @ margin_mask)/x.shape[0]
        # # return grad2/x.shape[0]

        grad = None
        # ====== YOUR CODE: ======
        g_dict = self.grad_ctx
        M, mask, x = g_dict['M'], g_dict['mask'], g_dict['x']
        g = (M > 0).float()
        g[mask==0] = 0
        g[torch.logical_not(mask)] = -g.sum(dim=1)
        grad = (x.t() @ g)/x.shape[0]
        return grad
