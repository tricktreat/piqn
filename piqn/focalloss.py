# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     r"""
#     """
#     def __init__(self, class_num, alpha, gamma=0, reduction = "none"):
#         super(FocalLoss, self).__init__()
#         # if alpha is None:
#         #     self._alpha = torch.ones(class_num, 1)
#         # else:
#         #     self._alpha = alpha
#         self._alpha = alpha
#         self.gamma = gamma
#         self.class_num = class_num
#         self.reduction = reduction

#     @property
#     def alpha(self):
#         return self._alpha

#     @alpha.setter
#     def alpha(self, alpha):
#         self._alpha = alpha

#     def forward(self, inputs, targets, alpha):
#         # import pdb; pdb.set_trace()
#         N = inputs.size(0)
#         C = inputs.size(1)
#         # P = F.softmax(inputs, dim=-1)
#         P = F.log_softmax(inputs, dim=-1)

#         class_mask = inputs.data.new(N, C).fill_(0).to(device=inputs.device)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)

#         # self._alpha = self._alpha.to(device=inputs.device)
#         alpha = alpha[ids.data.view(-1)]

#         log_p = (P*class_mask).sum(1).view(-1,1)

#         probs = log_p.exp()

#         # log_p = probs.log()

#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

#         loss = batch_loss
#         if self.reduction=="mean":
#             loss = batch_loss.mean()
#         if self.reduction=="sum":
#             loss = batch_loss.sum()
#         return loss


from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, class_num = 3, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha,torch.Tensor):
            assert len(alpha)==class_num   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self._alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self._alpha = torch.zeros(class_num)
            self._alpha[0] += alpha
            self._alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self._alpha = self._alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self._alpha = self._alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self._alpha, loss.t())
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss
