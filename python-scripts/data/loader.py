from data.dataset_ext import DataLoaderExt

from data.chen_example import ChenDataset
from data.silver_box import SilverBoxDataset


def load_dataset(dataset, dataset_options, train_batch_size, test_batch_size):
    if dataset == 'chen':
        loader_train = DataLoaderExt(ChenDataset(seq_len=dataset_options['seq_len'], **dataset_options['train']),
                                     batch_size=train_batch_size, shuffle=True, num_workers=4)
        loader_valid = DataLoaderExt(ChenDataset(seq_len=dataset_options['seq_len'], **dataset_options['valid']),
                                     batch_size=test_batch_size, shuffle=False, num_workers=4)
    elif dataset == 'silverbox':

        loader_train = DataLoaderExt(SilverBoxDataset(**dataset_options, split='train'),
                                     batch_size=train_batch_size, shuffle=False, num_workers=4)
        loader_valid = DataLoaderExt(SilverBoxDataset(**dataset_options, split='valid'),
                                     batch_size=test_batch_size, shuffle=False, num_workers=4)
    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    # Not implemented yet
    loader_test = None

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
