"""
Implement data generator for the example used in:

S. Chen, S. A. Billings, P. M. Grant, Non-Linear System Identification Using
Neural Networks, International Journal of Control 51 (6) (1990) 1191–
1214.
"""
# %% Prepare
import numpy as np
import numpy.random as rd


def nonlinear_function(y1, y2, u1, u2):
    return (0.8 - 0.5 * np.exp(-y1 ** 2)) * y1 - (0.3 + 0.9 * np.exp(-y1 ** 2)) * y2 \
           + u1 + 0.2 * u2 + 0.1 * u1 * u2


def generate_random_input(n, nrep, sd=1, seed=1):
    rd.seed(seed)
    u = sd*rd.randn(int(n//nrep))
    return np.repeat(u, nrep)


def simulate_system(u, sd_v, sd_w, seed=1):
    rd.seed(seed)
    n = np.shape(u)[0]
    v = sd_v * rd.randn(n)
    w = sd_w * rd.randn(n)
    y = rd.randn(n)
    for k in range(2, n):
        y[k] = nonlinear_function(y[k - 1], y[k - 2], u[k - 1], u[k - 2]) + v[k]
    return y + w


def chen_example(seq_len, n_batches):
    u = np.zeros((n_batches, 1, seq_len))
    y = np.zeros((n_batches, 1, seq_len))
    for i in range(n_batches):
        u[i, 0, :] = generate_random_input(seq_len, 5)
        y[i, 0, :] = simulate_system(u[i, 0, :], sd_v=0.1, sd_w=0.5)
    return u, y



