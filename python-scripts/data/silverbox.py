from pathlib import Path
import scipy.io
import numpy as np
import os
import urllib
import urllib.request
import zipfile
import numpy.random as rd
from data.base import IODataset


def create_silverbox_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None,
                              train_split=None, shuffle_seed=None):
    """Load silverbox data: train, validation and test datasets.

    Parameters
    ----------
    seq_len_train, seq_len_val, seq_len_test: int (optional)
        Maximum lenght for a batch on the training set. If `seq_len`
        is smaller than the total data length, the data will
        be further divided in batches. If None, put the entire dataset on
        a single batch.
    train_split: {None, int} (optional)
        Number of multisine realizations on the test set. Should be a number
        between 1 and 9. If `None` do not split the dataset and use all multisine
        realizations both for training and validation. Since there is very little
        noise on the data this make sense. By default it is None.
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
    u = mat['V1'][0]  # Input
    y = mat['V2'][0]  # Output

    # Number of samples of each subset of data
    n_zeros = 100  # Number of zeros at the start
    n_test = 40400  # Number of samples in the test set
    n_trans_before = 460  # Number of transient samples before each multisine realization
    n = 8192  # Number of samples per multisine realization
    n_trans_after = 40  # Number of transient samples after each multisine realization
    n_block = n_trans_before + n + n_trans_after
    n_multisine = 10  # Number of multisine realizations
    r_train = train_split if train_split is not None else n_multisine
    r_val = n_multisine - train_split if train_split is not None else n_multisine

    count = range(n_multisine)
    count = rd.permutation(count) if shuffle_seed is not None else count
    # Extract training data
    u_train = np.zeros(r_train * n)
    y_train = np.zeros(r_train * n)
    count_train = count[:train_split] if train_split is not None else count
    for i, r in enumerate(count_train):
        u_train[i * n + np.arange(n)] = u[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
        y_train[i * n + np.arange(n)] = y[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]

    # Extract validation data
    u_val = np.zeros(r_val * n)
    y_val = np.zeros(r_val * n)
    count_val = count[train_split:] if train_split is not None else count
    for i, r in enumerate(count_val):
        u_val[i * n + np.arange(n)] = u[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
        y_val[i * n + np.arange(n)] = y[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]

    # Extract test data
    u_test = u[n_zeros:n_zeros + n_test]
    y_test = y[n_zeros:n_zeros + n_test]

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test


def maybe_download_and_extract():
    """Download the data from nonlinear benchmark website, unless it's already here."""
    src_url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip'
    home = Path.home()
    work_dir = str(home.joinpath('datasets/SilverBox'))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    zipfilepath = os.path.join(work_dir, "SilverboxFiles.zip")
    if not os.path.exists(zipfilepath):
        filepath, _ = urllib.request.urlretrieve(
            src_url, zipfilepath)
        file = os.stat(filepath)
        size = file.st_size
        print('Successfully downloaded', 'SilverboxFiles.zip', size, 'bytes.')
    else:
        print('SilverboxFiles.zip', 'already downloaded!')

    datafilepath = os.path.join(work_dir, "SilverboxFiles/SNLS80mV.mat")
    print(datafilepath)
    if not os.path.exists(datafilepath):
        zip_ref = zipfile.ZipFile(zipfilepath, 'r')
        zip_ref.extractall(work_dir)
        zip_ref.close()
        print('Successfully unzipped data')
    return datafilepath


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # For testing purposes, should be removed afterwards
    train, val, test = create_silverbox_datasets()
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
