import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup

from utils import get_dataset
from hook import PolicyUpdateHook, TimeConsistencyHook
from diffaug import Augmenter

class AAAA(AlgorithmBase):
    """
        AdaAdvAutoAug algorithm.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.num_ops = 5

        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        
        self.warm_up_adv = 512
        self.bound = 0.002
        self.tau = 0.1
        self.policy_lam = 1.0
        self.aaaa_threshold = 0.9
        self.mem_update = Variable(torch.zeros(self.args.ulb_dest_len, dtype=torch.bool, requires_grad=False).cuda(self.gpu))
        self.mem_logits = Variable(torch.ones([self.args.ulb_dest_len, self.num_classes], dtype=torch.int64, requires_grad=False).cuda(self.gpu) + 0.01)
        # time consistency
        self.mem_tc = Variable(torch.zeros(self.args.ulb_dest_len, requires_grad=False).cuda(self.gpu))
        self.threshold = 1

        self.augmenter = self.set_augmenter()
        # self.policy, self.policy_optimizer, self.policy_scheduler = self.set_policy()
        self.policy, self.policy_optimizer = self.set_policy()
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(TimeConsistencyHook(), "TimeConsistencyHook")
        self.register_hook(PolicyUpdateHook(), "PolicyUpdateHook")
        super().set_hooks()

    def set_dataset(self):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir)
        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_policy(self):
        policy = self.net_builder(num_classes=len(self.augmenter.operations))
        optimizer = get_optimizer(policy)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter // self.args.d_steps,
                                                    self.args.num_warmup_iter)
        return policy, optimizer#, scheduler

    def set_augmenter(self):
        mean, std = self.dataset_dict['mean'], self.dataset_dict['std']
        augmenter = Augmenter(mean, std)
        return augmenter

    def apply_augmentation(self, x, mag):
        return self.augmenter(x, mag)
        
    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            self.call_hook("memory_update", "TimeConsistencyHook")

            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                _, targets_ulb_w = outs_x_ulb_w['logits'].max(dim=1)
                feats = self.normalize_flatten_features(outs_x_ulb_w['feats'])
                probs_x_ulb_w = F.softmax(outs_x_ulb_w['logits'] / self.T, dim=-1)
                y_ulb_w = torch.log(torch.gather(probs_x_ulb_w, 1, targets_ulb_w.view(-1, 1)).squeeze(dim=1))

                a_t = F.kl_div(self.mem_logits[idx_ulb].log(), probs_x_ulb_w, reduction='none').sum(dim=1)
                mask_smooth = (self.mem_tc[idx_ulb]).lt(self.threshold)

            train_policy = mask_smooth.sum() > 0 and self.it > self.warm_up_adv
            
            if train_policy:
                mag = self.policy(x_ulb_s)['logits'].sigmoid()

                indices = torch.argsort(torch.rand_like(mag), dim=-1)
                mask = torch.where(indices < self.num_ops, 1.0, 0.0).requires_grad_(False)
                mag = mag * mask

                x_adv = self.apply_augmentation(x_ulb_s[mask_smooth], mag[mask_smooth])

                outs_x_adv = self.model(x_adv)
                logits_adv, feats_adv = outs_x_adv['logits'], outs_x_adv['feats']
                probs_adv = torch.softmax(logits_adv / self.T, dim=-1)
                y_adv = torch.log(torch.gather(probs_adv, 1, targets_ulb_w[mask_smooth].view(-1, 1)).squeeze(dim=1))

                pip = (self.normalize_flatten_features(feats_adv) - feats[mask_smooth]).norm(dim=1).mean()
                constraint = y_ulb_w[mask_smooth] - y_adv
                policy_loss = -pip + self.policy_lam * F.relu(constraint - self.bound).mean()

                self.call_hook("policy_update", "PolicyUpdateHook", policy_loss)
            else:
                indices = torch.argsort(torch.rand(x_ulb_w.shape[0], len(self.augmenter.operations)), dim=-1)
                mask = torch.where(indices < self.num_ops, 1.0, 0.0).requires_grad_(False).to(x_ulb_w.device)

            mag = self.policy(x_ulb_s)['logits'].sigmoid()
            mag = mag * mask
            x_ulb_s = self.apply_augmentation(x_ulb_s, mag)

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

            # pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
            #                               logits=logits_x_ulb_w,
            #                               use_hard_label=False,
            #                               T=self.T,
            #                               softmax=True)
            # max_probs, targets_x_ulb_w = torch.max(pseudo_label, dim=-1)
            # mask_aaaa = (max_probs[mask_smooth].ge(self.aaaa_threshold)).float()

            # if train_policy:
            #     outs_x_ulb_s = self.model(x_ulb_s[mask_smooth])
            #     adv_loss = (F.cross_entropy(outs_x_ulb_s['logits'], targets_x_ulb_w[mask_smooth], reduction='none') * mask_aaaa).mean()
            #     total_loss += adv_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
        self.call_hook("policy_step", "PolicyUpdateHook")

        with torch.no_grad():
            self.mem_update[idx_ulb] = True
            self.mem_tc[idx_ulb] = 0.01 * self.mem_tc[idx_ulb] - 0.99 * a_t
            self.mem_logits[idx_ulb] = probs_x_ulb_w

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/policy_loss'] = policy_loss.item() if train_policy else 0
        # tb_dict['train/adv_loss'] = adv_loss.item() if train_policy else 0
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        tb_dict['train/mask_smooth_ratio'] = mask_smooth.float().mean().item()
        return tb_dict
    
    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'policy': self.policy.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        return save_dict

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
