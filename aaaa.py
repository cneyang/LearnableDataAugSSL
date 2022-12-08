import os
import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup

from utils import get_dataset
from hook import PolicyUpdateHook, DiscriminatorUpdateHook
from diffaug import Augmenter
import nets

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
        self.init(args = args, T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, lambda_p=args.policy_loss_ratio)
    
    def init(self, args, T, p_cutoff, hard_label=True, lambda_p=1.0):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.lambda_p = lambda_p
        self.suppression_loss = args.suppression_loss
        self.cutout = args.cutout

        self.augmenter = self.set_augmenter()
        self.policy, self.policy_optimizer, self.policy_scheduler = self.set_policy()

        if self.suppression_loss != 'none':
            self.lambda_sup = args.suppression_loss_ratio

        if self.suppression_loss == 'discriminator':
            self.discriminator, self.optimizer_D, self.criteria_D = self.set_discriminator()
        elif self.suppression_loss == 'perceptual':
            self.perceptual_loss = nets.WRNPerceptualLoss()
        
    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(PolicyUpdateHook(), "PolicyUpdateHook")

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
        return policy, optimizer, scheduler

    def set_augmenter(self):
        mean, std = self.dataset_dict['mean'], self.dataset_dict['std']
        augmenter = Augmenter(mean, std, cutout=self.cutout)
        return augmenter

    def apply_augmentation(self, x, mag, requires_grad=False):
        if requires_grad:
            return self.augmenter(x, mag)
        else:
            with torch.no_grad():
                return self.augmenter(x, mag)

    def set_discriminator(self):
        self.register_hook(DiscriminatorUpdateHook(), "DiscriminatorUpdateHook")
        discriminator = nets.Discriminator(channels=3, num_classes=self.args.num_classes, img_size=self.args.img_size)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr)
        criteria = torch.nn.BCELoss()
        return discriminator, optimizer, criteria

    def label_preserving_loss(self, x_ulb_w, x_ulb_s, targets, mask=None, sigma=0.002):
        probs_w = F.softmax(x_ulb_w, dim=-1)
        probs_s = F.softmax(x_ulb_s, dim=-1)
        log_probs_w = torch.log(torch.gather(probs_w, 1, targets.unsqueeze(1)))
        log_probs_s = torch.log(torch.gather(probs_s, 1, targets.unsqueeze(1)))
        loss = F.relu(log_probs_w - log_probs_s - sigma)
        if mask is not None:
            loss = loss * mask
        return loss.mean()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        if self.it % self.args.d_steps == 0:
            train_policy = True
            self.policy.train()
        else:
            train_policy = False
            self.policy.eval()

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                mag = self.policy(x_ulb_s)['logits']
            x_ulb_s_unaugmented = x_ulb_s
            x_ulb_s = self.apply_augmentation(x_ulb_s, mag, requires_grad=train_policy)

            if self.suppression_loss == 'perceptual':
                with torch.no_grad():
                    perceptual_loss = self.perceptual_loss(x_ulb_s, x_ulb_s_unaugmented)

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

            if self.suppression_loss == 'discriminator':
                batch_size = x_ulb_w.size(0)

                valid = torch.cuda.FloatTensor(batch_size, 1).fill_(1.0)
                fake = torch.cuda.FloatTensor(batch_size, 1).fill_(0.0)

                fake_pred, _ = self.discriminator(x_ulb_s)

                suppression_loss = self.criteria_D(fake_pred, valid)

                self.optimizer_D.zero_grad()

                fake_pred, _ = self.discriminator(x_ulb_s)
                real_pred, _ = self.discriminator(x_ulb_s_unaugmented)

                discriminator_loss = self.criteria_D(torch.cat((real_pred, fake_pred)), torch.cat((valid, fake)))

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

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        if train_policy:
            mag = self.policy(x_ulb_s)['logits']
            x_ulb_s = self.apply_augmentation(x_ulb_s, mag, requires_grad=train_policy)
            
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            logits_x_ulb_w = self.model(x_ulb_w)['logits'].detach()

            policy_loss = -self.lambda_p * consistency_loss(logits_x_ulb_s,
                                                            pseudo_label,
                                                            'ce',
                                                            mask=mask)
    
            if self.suppression_loss == 'discriminator':
                policy_loss += -self.lambda_sup * suppression_loss
            elif self.suppression_loss == 'perceptual':
                policy_loss += -self.lambda_sup * perceptual_loss
            elif self.suppression_loss == 'label_preserving':
                label_preserving_loss = self.label_preserving_loss(logits_x_ulb_w,
                                                       logits_x_ulb_s,
                                                       pseudo_label,
                                                       mask=mask)
                policy_loss += self.lambda_sup * label_preserving_loss
            self.call_hook("policy_update", "PolicyUpdateHook", loss=policy_loss)

            if self.suppression_loss == 'discriminator':
                self.call_hook("discriminator_update", "DiscriminatorUpdateHook", loss=discriminator_loss)

        
        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        if self.suppression_loss == 'discriminator':
            tb_dict['train/discriminator_loss'] = discriminator_loss.item()
            tb_dict['train/suppression_loss'] = suppression_loss.item()
        elif self.suppression_loss == 'perceptual':
            tb_dict['train/suppression_loss'] = perceptual_loss.item()
        elif self.suppression_loss == 'label_preeserving':
            tb_dict['train/suppression_loss'] = label_preserving_loss.item()

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

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
