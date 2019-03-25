from pathlib import Path
import scipy.io
import numpy as np
import os
import urllib
import urllib.request
import zipfile
import numpy.random as rd
from data.base import IODataset


def create_eeg_datasets(participant_idx=0, seq_len_train=None, seq_len_val=None, seq_len_test=None,
                              train_split=None, shuffle_seed=None):
    """Load eeg small data: train, validation and test datasets.

    Parameters
    ----------
    seq_len_train, seq_len_val, seq_len_test: int (optional)
        Maximum length for a batch on, respectively, the training,
        validation and test sets. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If None,
        put the entire dataset on a single batch.
    train_split: {None, int} (optional)
        Number of multisine realizations on the training set. Should be a number
        between 1 and 9. The remaining realizations `(10 - train_split)` will
        be used for validation. If `None` do not split the dataset and use all
        multisine realizations both for training and validation. Since there is
        very little noise on the data this is a reasonable choice and is used
        by default.
    shuffle_seed: {int, None}
        Seed used to shuffle the data between train and validation. If None, there is
        no shuffle at all.

    Returns
    -------
    dataset_train, dataset_val, dataset_test: Dataset
        Train, validation and test data

    Note
    ----
    Based on https://github.com/locuslab/TCN/blob/master/TCN/lambada_language/utils.py
    """
    # Extract input and output data Silverbox
    mat = scipy.io.loadmat(maybe_download_and_extract())

    n_participants = 10  # Number of participants
    n_realizations = 7
    seq_len = 256

    u = mat['data']['u'][0, 0]
    y = mat['data']['y'][0, 0]

    if train_split is None:
        train_split = n_realizations-2

    count = range(n_realizations-1)
    count = rd.permutation(count) if shuffle_seed is not None else count

    count_train = count[:train_split] if train_split is not None else count
    count_val = count[train_split:] if train_split is not None else count

    u_train = u[participant_idx, count_train, ...].reshape(-1)
    y_train = y[participant_idx, count_train, ...].reshape(-1)

    u_val = u[participant_idx, count_val, ...].reshape(-1)
    y_val = y[participant_idx, count_val, ...].reshape(-1)

    u_test = u[participant_idx, -1:, ...].reshape(-1)
    y_test = y[participant_idx, -1:, ...].reshape(-1)

    if seq_len_train is None:
        seq_len_train = seq_len
    else:
        assert seq_len % seq_len_train == 0

    if seq_len_val is None:
        seq_len_val = seq_len
    else:
        assert seq_len % seq_len_val == 0

    if seq_len_test is None:
        seq_len_test = seq_len
    else:
        assert seq_len % seq_len_test == 0

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test


def maybe_download_and_extract():
    """Download the data from nonlinear benchmark website, unless it's already here."""
    src_url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EEG/Benchmark_EEG_small.zip'
    home = Path.home()
    filename_download = "Benchmark_EEG_small.zip"
    work_dir = str(home.joinpath('datasets/eeg_data/small'))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    zipfilepath = os.path.join(work_dir, filename_download)
    if not os.path.exists(zipfilepath):
        print('Starting downloading', filename_download)
        filepath, _ = urllib.request.urlretrieve(
            src_url, zipfilepath)
        file = os.stat(filepath)
        size = file.st_size
        print('Successfully downloaded', filename_download, size, 'bytes.')
    else:
        print(filename_download, 'already downloaded!')

    datafilepath = os.path.join(work_dir, "Benchmark_EEG_small.mat")
    print(datafilepath)
    if not os.path.exists(datafilepath):
        print('Starting unzipping', filename_download)
        zip_ref = zipfile.ZipFile(zipfilepath, 'r')
        zip_ref.extractall(work_dir)
        zip_ref.close()
        print('Successfully unzipped data')
    return datafilepath


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # For testing purposes, should be removed afterwards
    train, val, test = create_eeg_datasets()
    # Convert back from torch tensor to numpy vector
    u_train = train.u.reshape(-1)
    y_train = train.y.reshape(-1)
    k_train = 1 + np.arange(len(u_train))
    u_val = val.u.reshape(-1)
    y_val = val.y.reshape(-1)
    k_val = 1 + np.arange(len(u_val))
    u_test = test.u.reshape(-1)
    y_test = test.y.reshape(-1)
    k_test = 1 + np.arange(len(u_test))
    # Plot training data
    plt.figure()
    plt.plot(k_train, u_train)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input training data')
    plt.show()
    plt.figure()
    plt.plot(k_train, y_train)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output training data')
    plt.show()
    # Plot validation data
    plt.figure()
    plt.plot(k_val, u_val)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input validation data')
    plt.show()
    plt.figure()
    plt.plot(k_val, y_val)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output validation data')
    plt.show()
    # Plot test data
    plt.figure()
    plt.plot(k_test, u_test)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input test data')
    plt.show()
    plt.figure()
    plt.plot(k_test, y_test)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output test data')
    plt.show()