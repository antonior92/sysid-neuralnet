import scipy.io
import numpy as np
#import matplotlib.pyplot as plt

def data_generator(args):
    # Extract input and output data
    mat = scipy.io.loadmat('../data/SilverboxFiles/SNLS80mV.mat')
    X = mat['V1'][0] # Input
    Y = mat['V2'][0] # Output
    
    # Number of samples of each subset of data
    Nzeros = 100; # Number of zeros at the start
    Ntest = 40400 # Number of samples in the test set
    NtransBefore = 460 # Number of transient samples before each realization
    N = 8192 # Number of samples per realization
    NtransAfter = 40 # Number of transient samples after each realization
    Nblock = NtransBefore + N + NtransAfter
    R = 9 # Number of multisine realizations in test set
    Rval = 10 - R # Number of multisine realizations in validation set
    
    # Extract training data
    X_train = np.zeros(R * N)
    Y_train = np.zeros(R * N)
    for r in range(R):
        X_train[r * N + np.arange(N)] = X[Nzeros + Ntest + r * Nblock + NtransBefore + np.arange(N)]
        Y_train[r * N + np.arange(N)] = Y[Nzeros + Ntest + r * Nblock + NtransBefore + np.arange(N)]
    
    # Extract validation data
    X_val = np.zeros(Rval * N)
    Y_val = np.zeros(Rval * N)
    for r in range(Rval):
        X_val[r * N + np.arange(N)] = X[Nzeros + Ntest + (R + r) * Nblock + NtransBefore + np.arange(N)]
        Y_val[r * N + np.arange(N)] = Y[Nzeros + Ntest + (R + r) * Nblock + NtransBefore + np.arange(N)]
    
    # Extract test data
    X_test = X[Nzeros:Nzeros + Ntest]
    Y_test = Y[Nzeros:Nzeros + Ntest]
    
    # Still need to split in (nBatches,nInputsorOutputs,nSamples)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test;

## For testing purposes, should be removed afterwards
#args = {'batch_size': 1}
#X_train, Y_train, X_val, Y_val, X_test, Y_test = data_generator(args)
## Plot training data
#plt.figure()
#plt.plot(1 + np.arange(len(X_train)), X_train)
#plt.xlabel('Sample number')
#plt.ylabel('Input (V)')
#plt.title('Input training data')
#plt.show()
#plt.figure()
#plt.plot(1 + np.arange(len(Y_train)), Y_train)
#plt.xlabel('Sample number')
#plt.ylabel('Output (V)')
#plt.title('Output training data')
#plt.show()
## Plot validation data
#plt.figure()
#plt.plot(1 + np.arange(len(X_val)), X_val)
#plt.xlabel('Sample number')
#plt.ylabel('Input (V)')
#plt.title('Input validation data')
#plt.show()
#plt.figure()
#plt.plot(1 + np.arange(len(Y_val)), Y_val)
#plt.xlabel('Sample number')
#plt.ylabel('Output (V)')
#plt.title('Output validation data')
#plt.show()
## Plot test data
#plt.figure()
#plt.plot(1 + np.arange(len(X_test)), X_test)
#plt.xlabel('Sample number')
#plt.ylabel('Input (V)')
#plt.title('Input test data')
#plt.show()
#plt.figure()
#plt.plot(1 + np.arange(len(Y_test)), Y_test)
#plt.xlabel('Sample number')
#plt.ylabel('Output (V)')
#plt.title('Output test data')
#plt.show()
