from data.dataset_ext import DataLoaderExt

from data.chen_example import ChenDataset
from data.silver_box import create_silverbox_datasets


def load_dataset(dataset, dataset_options, train_batch_size, test_batch_size):
    if dataset == 'chen':
        loader_train = DataLoaderExt(ChenDataset(seq_len=dataset_options['seq_len'], **dataset_options['train']),
                                     batch_size=train_batch_size, shuffle=True, num_workers=4)
        loader_valid = DataLoaderExt(ChenDataset(seq_len=dataset_options['seq_len'], **dataset_options['valid']),
                                     batch_size=test_batch_size, shuffle=False, num_workers=4)
        loader_test = DataLoaderExt(ChenDataset(seq_len=dataset_options['seq_len'], **dataset_options['test']),
                                     batch_size=test_batch_size, shuffle=False, num_workers=4)
    elif dataset == 'silverbox':

        dataset_train, dataset_valid, dataset_test = create_silverbox_datasets(**dataset_options)

        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=4)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=4)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=4)

    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
