# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool

from hook import AdversarialAttackHook, TimeConsistencyHook


class LPA3(AlgorithmBase):
    """
        Label-Preserving Adversarial Auto-Augment algorithm (https://arxiv.org/pdf/2211.00824.pdf).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - warm_up_adv (`int`, *optional*, defaults to 5):
            Warm up epoch for LPA3
        - 
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # lap3 specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

        self.warm_up_adv = 5
        self.bound = 0.1
        self.tau = 0.1
        self.attack_iters = 5
        self.attack_lam = 0.1
        self.lpa3_threshold = 0.9
        self.mem_update = Variable(torch.zeros(self.args.ulb_dest_len, dtype=torch.bool, requires_grad=False).cuda(self.gpu))
        self.mem_logits = Variable(torch.ones([self.args.ulb_dest_len, self.num_classes], dtype=torch.int64, requires_grad=False).cuda(self.gpu) + 0.01)
        # time consistency
        self.mem_tc = Variable(torch.zeros(self.args.ulb_dest_len, requires_grad=False).cuda(self.gpu))
        self.threshold = 1

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(AdversarialAttackHook(), "AdversarialAttackHook")
        self.register_hook(TimeConsistencyHook(), "TimeConsistencyHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            self.call_hook("memory_update", "TimeConsistencyHook")
            
            # adversarial training
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w, adv=True)
                _, targets_ulb_w  = torch.max(outs_x_ulb_w['logits'], dim=1)
                feats = self.normalize_flatten_features(outs_x_ulb_w['feats'])
                probs_x_ulb_w = F.softmax(outs_x_ulb_w['logits'] / self.T, dim=-1)
                y_ulb_w = torch.log(torch.gather(probs_x_ulb_w, 1, targets_ulb_w.view(-1, 1)).squeeze(dim=1))
                # adversarial training
                a_t = F.kl_div(self.mem_logits[idx_ulb].log(), probs_x_ulb_w, reduction='none').sum(dim=1)
                mask_smooth = (self.mem_tc[idx_ulb]).lt(self.threshold)

            run_adv = mask_smooth.sum() > 0
            train_adv = run_adv and self.epoch >= self.warm_up_adv

            if run_adv:
                adv_inputs = self.call_hook("attack", "AdversarialAttackHook", x_ulb_w[mask_smooth], targets_ulb_w[mask_smooth], y_ulb_w[mask_smooth], feats[mask_smooth])

            # fixmatch
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=False,
                                          T=self.T,
                                          softmax=True)
            max_probs, targets_x_ulb_w = torch.max(pseudo_label, dim=1)
            mask_lpa3 = (max_probs[mask_smooth].ge(self.lpa3_threshold)).float()

            if run_adv:
                outs_x_adv = self.model(adv_inputs, adv=True)
                l_adv = (F.cross_entropy(outs_x_adv['logits'], targets_x_ulb_w[mask_smooth], reduction='none') * mask_lpa3).mean()

            if train_adv:
                total_loss += l_adv

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        with torch.no_grad():
            self.mem_update[idx_ulb] = True
            self.mem_tc[idx_ulb] = 0.01 * self.mem_tc[idx_ulb] - 0.99 * a_t
            self.mem_logits[idx_ulb] = probs_x_ulb_w

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/adv_loss'] = l_adv.item() if run_adv else 0
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict
    
    def normalize_flatten_features(self, features, eps=1e-10):
        normalized_features = []
        for feature_layer in features:
            norm_factor = torch.sqrt(torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
            normalized_features.append(
                (feature_layer / (norm_factor *
                                np.sqrt(feature_layer.size()[2] *
                                        feature_layer.size()[3])))
                .view(feature_layer.size()[0], -1)
            )
        return torch.cat(normalized_features, dim=1)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
