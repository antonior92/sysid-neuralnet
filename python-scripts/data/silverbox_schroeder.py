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
    # Load Schroeder multisine data
    mat_schroeder = scipy.io.loadmat(maybe_download_and_extract_schroeder())
    u_schroeder = mat_schroeder['V1'][0]  # Input
    y_schroeder = mat_schroeder['V2'][0]  # Output

    # Remove offset errors
    u_schroeder = u_schroeder - np.mean(u_schroeder)
    y_schroeder = y_schroeder - np.mean(y_schroeder)

    # Select the Schroeder section of the experiment
    u_schroeder = u_schroeder[10584 + np.arange(1201)]
    y_schroeder = y_schroeder[10584 + np.arange(1201)]

    # Use these data as test data
    u_test = u_schroeder
    y_test = y_schroeder

    # Load estimation data
    mat = scipy.io.loadmat(maybe_download_and_extract())
    u = mat['V1'][0]  # Input
    y = mat['V2'][0]  # Output

    # Remove offset errors
    u = u - np.mean(u)
    y = y - np.mean(y)

    # Select 10 blocks of 100+400+8192 samples in the multisine bit of the data
    n_zeros = 100  # Number of zeros at the start
    n_test = 40400  # Number of samples in the test set (arrow data)
    n_trans_before = 460  # Number of transient samples before each multisine realization
    n = 8192  # Number of samples per multisine realization
    n_trans_after = 40  # Number of transient samples after each multisine realization
    n_block = n_trans_before + n + n_trans_after
    n_multisine = 10  # Number of multisine realizations
    u = u[n_zeros + n_test + np.arange(n_multisine * n_block)]
    y = y[n_zeros + n_test + np.arange(n_multisine * n_block)]
    # Remove means
    u = u - np.mean(u)
    y = y - np.mean(y)

    # Eliminate zeros and transients (not used later on?)
    uP = np.zeros(n_multisine * n)
    yP = np.zeros(n_multisine * n)
    for i in range(n_multisine):
        uP[i * n + np.arange(n)] = u[i * n_block + n_trans_before + np.arange(n)]
        yP[i * n + np.arange(n)] = y[i * n_block + n_trans_before + np.arange(n)]

    # Select first block as estimation data
    uAll = u
    yAll = y
    u_train = u[np.arange(n_block)]
    y_train = y[np.arange(n_block)]

    # No splitting in estimation and validation data
    u_val = u_train
    y_val = y_train

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


def maybe_download_and_extract_schroeder():
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

    datafilepath = os.path.join(work_dir, "SilverboxFiles/Schroeder80mV.mat")
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