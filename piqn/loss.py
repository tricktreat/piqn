from abc import ABC
from pdb import set_trace

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch._C import dtype
from .matcher import HungarianMatcher
from .focalloss import FocalLoss, sigmoid_focal_loss
import torchsnooper


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class PIQNLoss(Loss):
    def __init__(self, entity_type_count, device, model, optimizer, scheduler, max_grad_norm, nil_weight, match_class_weight, match_boundary_weight, loss_class_weight, loss_boundary_weight, type_loss, solver, match_warmup_epoch = 0):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        
        # losses = ['labels', 'boundary', 'cardinality']
        losses = ['labels', 'boundary']
        self.weight_dict = {'loss_ce': loss_class_weight, 'loss_boundary': loss_boundary_weight}
        self.criterion = Criterion(entity_type_count, self.weight_dict, nil_weight, losses, type_loss = type_loss, match_class_weight = match_class_weight, match_boundary_weight = match_boundary_weight, solver = solver, match_warmup_epoch = match_warmup_epoch)
        self.criterion.to(device)
        self._max_grad_norm = max_grad_norm

    def del_attrs(self):
        del self._optimizer 
        del self._scheduler

    def compute(self, entity_logits, pred_left, pred_right, output, gt_types, gt_spans, entity_masks, epoch, deeply_weight = "same", seq_logits = None, gt_seq_labels = None, batch = None):
        # set_trace()

        maskedlm_loss = None
        if seq_logits is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            maskedlm_loss = loss_fct(seq_logits.view(-1, seq_logits.size(-1)), gt_seq_labels.view(-1))
            
        

        gt_types_wo_nil = gt_types.masked_select(entity_masks)
        
        # if len(gt_types_wo_nil) == 0:
        #     return 0.1

        sizes = [i.sum() for i in entity_masks]
        entity_masks = entity_masks.unsqueeze(2).repeat(1, 1, 2)
        spans_wo_nil = gt_spans.masked_select(entity_masks).view(-1, 2)

        targets = {"labels": gt_types_wo_nil, "gt_left":spans_wo_nil[:, 0], "gt_right":spans_wo_nil[:, 1], "sizes":sizes}

        train_loss = []
        for out_dict in output:
            entity_logits, pred_left, pred_right = out_dict["entity_logits"], out_dict["p_left"], out_dict["p_right"]
            outputs = {"pred_logits":entity_logits, "pred_left":pred_left, "pred_right":pred_right, "token_mask": batch["token_masks"]}
            loss_dict = self.criterion(outputs, targets, epoch)
            
            train_loss.append(sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys()))
        if deeply_weight == "same":
            deeply_weight = [1] * len(output)
        elif deeply_weight == "linear":
            deeply_weight = list(range(1, len(output)+1))
        train_loss = sum(train_loss[i] * deeply_weight[i] for i in range(len(output)))

        if maskedlm_loss is not None:
            train_loss += maskedlm_loss
        train_loss.backward()
        # find unused parameters
        # for name, param in (self._model.named_parameters()):
        #     if param.grad is None:
        #         print(name)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()


class Criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, entity_type_count, weight_dict, nil_weight, losses, type_loss, match_class_weight, match_boundary_weight, solver, match_warmup_epoch):
        """ Create the criterion.
        Parameters:
            entity_type_count: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            nil_weight: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.entity_type_count = entity_type_count
        self.matcher = HungarianMatcher(cost_class = match_class_weight, cost_span = match_boundary_weight, solver = solver)
        self.match_warmup_epoch = match_warmup_epoch
        if match_warmup_epoch > 0:
            self.order_matcher = HungarianMatcher(solver = "order")
        self.weight_dict = weight_dict
        self.nil_weight = nil_weight
        self.losses = losses
        empty_weight = torch.ones(self.entity_type_count)
        empty_weight[0] = self.nil_weight
        self.register_buffer('empty_weight', empty_weight)
        self.type_loss = type_loss
        if type_loss == "focalloss":
            self.focalloss = FocalLoss(entity_type_count, alpha=empty_weight, gamma=2, reduction = "mean")

    def loss_labels(self, outputs, targets, indices, num_spans):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.entity_type_count,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o


        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        # indices [(tensor([44]), tensor([0])), (tensor([20, 42, 46, 56, 58]), tensor([2, 0, 3, 4, 1])), (tensor([ 7,  8, 11, 27]), tensor([0, 1, 3, 2])), (tensor([ 2, 14, 31, 39, 41, 43, 50]), tensor([6, 5, 4, 3, 0, 2, 1])), (tensor([45]), tensor([0])), (tensor([14, 18]), tensor([1, 0])), (tensor([3]), tensor([0])), (tensor([44, 47, 55]), tensor([2, 0, 1]))]
        
        # self._get_src_permutation_idx(indices) (tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7]), tensor([44, 20, 42, 46, 56, 58,  7,  8, 11, 27,  2, 14, 31, 39, 41, 43, 50, 45, 14, 18,  3, 44, 47, 55]))
        # self._get_tgt_permutation_idx(indices) (tensor([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7]), tensor([0, 2, 0, 3, 4, 1, 0, 1, 3, 2, 6, 5, 4, 3, 0, 2, 1, 0, 1, 0, 0, 2, 0, 1]))
        
        # targets["labels"] (tensor([5], device='cuda:2'), tensor([2, 2, 2, 2, 2], device='cuda:2'), tensor([2, 5, 5, 5], device='cuda:2'), tensor([2, 5, 2, 2, 3, 6, 3], device='cuda:2'), tensor([2], device='cuda:2'), tensor([4, 6], device='cuda:2'), tensor([2], device='cuda:2'), tensor([5, 2, 2], device='cuda:2'))

        # target_classes_o tensor([5, 2, 2, 2, 2, 2, 2, 5, 5, 5, 3, 6, 3, 2, 2, 2, 5, 2, 6, 4, 2, 2, 5, 2], device='cuda:2')

        labels = targets["labels"].split(targets["sizes"], dim=-1)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        empty_weight = self.empty_weight.clone()

        if self.nil_weight == -1:
            empty_weight[0] = num_spans / (src_logits.size(0) * src_logits.size(1) - num_spans)
            # empty_weight[0] = num_spans / (src_logits.size(0) * src_logits.size(1))
            # empty_weight[1:] = (src_logits.size(0) * src_logits.size(1) - num_spans) / (src_logits.size(0) * src_logits.size(1))
        if self.type_loss == "celoss":
            src_logits = src_logits.view(-1, src_logits.size(2))
            target_classes = target_classes.view(-1)
            loss_ce = F.cross_entropy(src_logits, target_classes, empty_weight)
            # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        if self.type_loss == "bceloss":
            src_logits = src_logits.view(-1, src_logits.size(2))
            target_classes = target_classes.view(-1)
            target_classes_onehot = torch.zeros([target_classes.size(0), src_logits.size(1)], dtype=torch.float32).to(device=target_classes.device)
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(1), 1)
            src_logits_p = F.sigmoid(src_logits)
            loss_ce = F.binary_cross_entropy(src_logits_p, target_classes_onehot)
        if self.type_loss == "focalloss":
            self.focalloss.alpha = empty_weight
            loss_ce = self.focalloss(src_logits.view(-1, src_logits.size(-1)), target_classes.view(-1))
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_spans):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        # tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        tgt_lengths = torch.as_tensor(targets["sizes"], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boundary(self, outputs, targets, indices, num_spans):
        idx = self._get_src_permutation_idx(indices)
        src_spans_left = outputs['pred_left'][idx]
        src_spans_right = outputs['pred_right'][idx]
        token_masks = outputs['token_mask'].unsqueeze(1).expand(-1, outputs['pred_right'].size(1), -1)
        token_masks = token_masks[idx]

        # target_spans_left = torch.cat([t['gt_left'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_spans_right = torch.cat([t['gt_right'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        gt_left = targets["gt_left"].split(targets["sizes"], dim=0)
        target_spans_left = torch.cat([t[i] for t, (_, i) in zip(gt_left , indices)], dim=0)
        gt_right = targets["gt_right"].split(targets["sizes"], dim=0)
        target_spans_right = torch.cat([t[i] for t, (_, i) in zip(gt_right , indices)], dim=0)


        # src_left_boundary_logp = torch.log(1e-25 + src_spans_left)
        # src_right_boundary_logp = torch.log(1e-25 + src_spans_right)


        # left_nll_loss = F.nll_loss(src_left_boundary_logp, target_spans_left, reduction='none')
        # right_nll_loss = F.nll_loss(src_right_boundary_logp, target_spans_right, reduction='none')

        # loss_boundary = left_nll_loss + right_nll_loss

        left_onehot = torch.zeros([target_spans_left.size(0), src_spans_left.size(1)], dtype=torch.float32).to(device=target_spans_left.device)
        left_onehot.scatter_(1, target_spans_left.unsqueeze(1), 1)
    
        right_onehot = torch.zeros([target_spans_right.size(0), src_spans_right.size(1)], dtype=torch.float32).to(device=target_spans_right.device)
        right_onehot.scatter_(1, target_spans_right.unsqueeze(1), 1)

        left_nll_loss = F.binary_cross_entropy(src_spans_left, left_onehot, reduction='none')
        right_nll_loss = F.binary_cross_entropy(src_spans_right, right_onehot, reduction='none')

        # left_nll_loss = sigmoid_focal_loss(src_spans_left, left_onehot)
        # right_nll_loss = sigmoid_focal_loss(src_spans_right, right_onehot)


        # NIL object boundary
        loss_boundary = (left_nll_loss + right_nll_loss) * token_masks

        losses = {}
        losses['loss_boundary'] = loss_boundary.sum() / num_spans
        # losses['loss_boundary'] = loss_boundary.mean(1).sum() / num_spans
        

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boundary': self.loss_boundary,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    # @torchsnooper.snoop()
    def forward(self, outputs, targets, epoch):
        # Retrieve the matching between the outputs of the last layer and the targets
        
        if epoch < self.match_warmup_epoch:
            indices = self.order_matcher(outputs, targets)
        else:
            indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_spans = sum(targets["sizes"])
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))
        return losses