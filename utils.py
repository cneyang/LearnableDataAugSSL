import nets

def get_net_builder(net_name, from_name: bool):
    return NotImplementedError

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
    from datasets import get_cifar, get_svhn, get_stl10

    if dataset in ["cifar10", "cifar100"]:
        lb_dset, ulb_dset, eval_dset = get_cifar(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'svhn':
        lb_dset, ulb_dset, eval_dset = get_svhn(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    elif dataset == 'stl10':
        lb_dset, ulb_dset, eval_dset = get_stl10(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        test_dset = None
    else:
        raise NotImplementedError

    dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
    return dataset_dict