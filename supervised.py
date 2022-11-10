# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.algorithms.utils import ce_loss
from utils import get_dataset


class Supervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def set_dataset(self):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels,
                                   self.args.num_classes, self.args.data_dir)
        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn(
            "unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)['logits']

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=sup_loss)

        # tensorboard_dict update
        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        return tb_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
            
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")
