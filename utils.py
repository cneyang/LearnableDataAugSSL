from dataset import get_cifar, get_svhn, get_stl10


def get_dataset(args, algorithm, dataset, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """
    create dataset

    Args
        args: argparse arguments
        algorithm: algrorithm name, used for specific return items in __getitem__ of datasets
        dataset: dataset name 
        num_labels: number of labeled data in dataset
        num_classes: number of classes
        seed: random seed
        data_dir: data folder
    """

    if dataset in ["cifar10", "cifar100"]:
        lb_dset, ulb_dset, eval_dset, mean, std = get_cifar(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'svhn':
        lb_dset, ulb_dset, eval_dset, mean, std = get_svhn(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'stl10':
        lb_dset, ulb_dset, eval_dset, mean, std = get_stl10(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    else:
        raise NotImplementedError

    dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset, 'mean': mean, 'std': std}
    return dataset_dict

from semilearn.algorithms import *
from aaaa import AAAA
from modernvat import ModernVAT
from lpa3 import LPA3

name2alg = {
    'fullysupervised': FullySupervised,
    'supervised': FullySupervised,
    'fixmatch': FixMatch,
    'flexmatch': FlexMatch,
    'adamatch': AdaMatch,
    'pimodel': PiModel,
    'meanteacher': MeanTeacher,
    'pseudolabel': PseudoLabel,
    'uda': UDA,
    'vat': VAT,
    'mixmatch': MixMatch,
    'remixmatch': ReMixMatch,
    'crmatch': CRMatch,
    'comatch': CoMatch,
    'simmatch': SimMatch,
    'dash': Dash,
    'aaaa': AAAA,
    'modernvat': ModernVAT,
    'lpa3': LPA3
}

def get_algorithm(args, net_builder, tb_log, logger):
    try:
        alg = name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    except KeyError as e:
        print(f'Unknown algorithm: {str(e)}')
