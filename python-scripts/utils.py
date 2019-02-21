import scipy.io
import numpy as np
import torch

# based on https://github.com/locuslab/TCN/blob/master/TCN/lambada_language/utils.py

def data_generator(args):
    # Extract input and output data Silverbox
    mat = scipy.io.loadmat('../data/SilverboxFiles/SNLS80mV.mat')
    U = mat['V1'][0] # Input
    Y = mat['V2'][0] # Output
    
    # Number of samples of each subset of data
    Nzeros = 100; # Number of zeros at the start
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
    
    # Turn into torch tensors
    U_train = torch.from_numpy(U_train)
    Y_train = torch.from_numpy(Y_train)
    U_val = torch.from_numpy(U_val)
    Y_val = torch.from_numpy(Y_val)
    U_test = torch.from_numpy(U_test)
    Y_test = torch.from_numpy(Y_test)
    
    # Reshape into (nBatches,nInputsorOutputs,nSamples)
    U_train = batchify(U_train, args['batch_size'], args)
    Y_train = batchify(Y_train, args['batch_size'], args)
    U_val = batchify(U_val, args['batch_size'], args)
    Y_val = batchify(Y_val, args['batch_size'], args)
    U_test = batchify(U_test, args['batch_size'], args)
    Y_test = batchify(Y_test, args['batch_size'], args)
    
    return U_train, Y_train, U_val, Y_val, U_test, Y_test;

def batchify(data, batch_size, args):
    # data should be a torch tensor
    # data should have size (total number of samples) times (number of signals)
    # The output has size (number of batches) times (number of signals) times (batch size)
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
#    data = data[range(nbatch * batch_size), :]
#    data = np.reshape(data, [batch_size, nbatch, -1], order='F')
#    data = np.transpose(data, (1, 2, 0))
    # Evenly divide the data across the batch_size batches and make sure it is still in temporal order
    data = data.view(nbatch, batch_size, -1).transpose(0, 1)
#    Transforming to cuda() is done in evaluate script
#    if args['cuda']:
#        data = data.cuda()
    return data;

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # For testing purposes, should be removed afterwards
    args = {'batch_size': 1}
    U_train, Y_train, U_val, Y_val, U_test, Y_test = data_generator(args)
    # Convert back from torch tensor to numpy vector
    X_train = U_train.numpy().reshape(-1)
    Y_train = Y_train.numpy().reshape(-1)
    X_val = U_val.numpy().reshape(-1)
    Y_val = Y_val.numpy().reshape(-1)
    X_test = U_test.numpy().reshape(-1)
    Y_test = Y_test.numpy().reshape(-1)
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
