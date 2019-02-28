from pathlib import Path
import scipy.io
import numpy as np
import os
import urllib
import urllib.request
import zipfile

from data.dataset_ext import DatasetExt


# based on https://github.com/locuslab/TCN/blob/master/TCN/lambada_language/utils.py

def create_silverbox_datasets(seq_len, seq_len_eval):
    U_train, Y_train, U_val, Y_val, U_test, Y_test = load_silverbox_data(seq_len, seq_len_eval)

    dataset_train = SilverBoxDataset(U_train, Y_train)
    dataset_val = SilverBoxDataset(U_val, Y_val)
    dataset_test = SilverBoxDataset(U_test, Y_test)

    return dataset_train, dataset_val, dataset_test

class SilverBoxDataset(DatasetExt):

    def __init__(self, u, y):
        self.u = u.astype(np.float32)
        self.y = y.astype(np.float32)
        self.ntotbatch = self.u.shape[0]
        self.seq_len = self.u.shape[2]

    @property
    def data_shape(self):
        return (1, self.seq_len), (1, self.seq_len)

    def __len__(self):
        return self.ntotbatch

    def __getitem__(self, idx):
        return self.u[idx, ...], self.y[idx, ...]


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


def load_silverbox_data(seq_len, seq_len_eval):
    # Extract input and output data Silverbox
    mat = scipy.io.loadmat(maybe_download_and_extract())

    U = mat['V1'][0] # Input
    Y = mat['V2'][0] # Output
    
    # Number of samples of each subset of data
    Nzeros = 100  # Number of zeros at the start
    Ntest = 40400 # Number of samples in the test set
    NtransBefore = 460 # Number of transient samples before each multisine realization
    N = 8192 # Number of samples per multisine realization
    NtransAfter = 40 # Number of transient samples after each multisine realization
    Nblock = NtransBefore + N + NtransAfter
    R = 9 # Number of multisine realizations in test set
    Rval = 10 - R # Number of multisine realizations in validation set
    
    # Extract training data
    U_train = np.zeros(R * N)
    Y_train = np.zeros(R * N)
    for r in range(R):
        U_train[r * N + np.arange(N)] = U[Nzeros + Ntest + r * Nblock + NtransBefore + np.arange(N)]
        Y_train[r * N + np.arange(N)] = Y[Nzeros + Ntest + r * Nblock + NtransBefore + np.arange(N)]
    
    # Extract validation data
    U_val = np.zeros(Rval * N)
    Y_val = np.zeros(Rval * N)
    for r in range(Rval):
        U_val[r * N + np.arange(N)] = U[Nzeros + Ntest + (R + r) * Nblock + NtransBefore + np.arange(N)]
        Y_val[r * N + np.arange(N)] = Y[Nzeros + Ntest + (R + r) * Nblock + NtransBefore + np.arange(N)]
    
    # Extract test data
    U_test = U[Nzeros:Nzeros + Ntest]
    Y_test = Y[Nzeros:Nzeros + Ntest]

    # Reshape into (nBatches,nInputsorOutputs, nSamples)
    U_train = batchify(U_train, seq_len)
    Y_train = batchify(Y_train, seq_len)
    if seq_len_eval is not None:
        U_val = batchify(U_val, seq_len_eval)
        Y_val = batchify(Y_val, seq_len_eval)
        U_test = batchify(U_test, seq_len_eval)
        Y_test = batchify(Y_test, seq_len_eval)
    else:
        U_val = batchify(U_val, U_val.shape[0])
        Y_val = batchify(Y_val, Y_val.shape[0])
        U_test = batchify(U_test, U_test.shape[0])
        Y_test = batchify(Y_test, Y_test.shape[0])

    return U_train, Y_train, U_val, Y_val, U_test, Y_test


def batchify(data, seq_len):
    """

    :param data: Data which is to be split up in batches
    :param seq_len: Length of each sequence in the batches
    :return: Batchified data
    """
    # data should be a torch tensor
    # data should have size (total number of samples) times (number of signals)
    # The output has size (number of batches) times (number of signals) times (batch size)
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.shape[0] // seq_len
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * seq_len]
#    data = data[range(nbatch * batch_size), :]
#    data = np.reshape(data, [batch_size, nbatch, -1], order='F')
#    data = np.transpose(data, (1, 2, 0))
    # Evenly divide the data across the batch_size batches and make sure it is still in temporal order
#    data = data.reshape((nbatch, 1, seq_len)).transpose(0, 1, 2)
    data = data.reshape((seq_len, nbatch, 1), order='F').transpose(1,2,0)
    #data = data.view(nbatch, batch_size, -1).transpose(0, 1)
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # For testing purposes, should be removed afterwards
    U_train, Y_train, U_val, Y_val, U_test, Y_test = load_silverbox_data(seq_len=1000)
    # Convert back from torch tensor to numpy vector
    X_train = U_train.reshape(-1)
    Y_train = Y_train.reshape(-1)
    X_val = U_val.reshape(-1)
    Y_val = Y_val.reshape(-1)
    X_test = U_test.reshape(-1)
    Y_test = Y_test.reshape(-1)
    # Plot training data
    plt.figure()
    plt.plot(1 + np.arange(len(X_train)), X_train)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input training data')
    plt.show()
    plt.figure()
    plt.plot(1 + np.arange(len(Y_train)), Y_train)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output training data')
    plt.show()
    # Plot validation data
    plt.figure()
    plt.plot(1 + np.arange(len(X_val)), X_val)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input validation data')
    plt.show()
    plt.figure()
    plt.plot(1 + np.arange(len(Y_val)), Y_val)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output validation data')
    plt.show()
    # Plot test data
    plt.figure()
    plt.plot(1 + np.arange(len(X_test)), X_test)
    plt.xlabel('Sample number')
    plt.ylabel('Input (V)')
    plt.title('Input test data')
    plt.show()
    plt.figure()
    plt.plot(1 + np.arange(len(Y_test)), Y_test)
    plt.xlabel('Sample number')
    plt.ylabel('Output (V)')
    plt.title('Output test data')
    plt.show()
