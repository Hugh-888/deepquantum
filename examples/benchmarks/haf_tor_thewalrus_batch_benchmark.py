
import argparse
import numpy as np
import time
import torch
import thewalrus

from thewalrus import hafnian, tor, loop_hafnian, ltor
from thewalrus.random import random_covariance
from thewalrus.quantum.conversions import Amat, Xmat

# Using TheWalrus to calculate the hafnian and torontonian of random matrix.
# This script performs hafnian and torontonian calculation with specified matrix size, batch sizes, and repeat times.
# It measures the time taken for each process.

# Example usage:
# plain hafnian
# python haf_tor_thewalrus_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2
# loop hafnian
# python haf_tor_thewalrus_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2 --haf_loop
# torontonian
# python haf_tor_thewalrus_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2
# loop torontonian
# python haf_tor_thewalrus_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2 --tor_loop

# - haf_size: the max size of matrix for hafnian (2, 4, ..., haf_size). Default: ``None``
# - haf_loop: Whether to calculate the loop hafnian. Default: ``False``
# - tor_loop: Whether to calculate the loop torontonian. Default: ``False``
# - tor_size: the max size of matrix for torontonian (2, 4, ..., tor_size). Default: ``None``
# - batch: Number of matrices to process for each calculation. Default: 1
# - repeat: Number of times to repeat the process. Default: 1
# The time taken for each process is recorded and saved in ".npy" file.

def generate_symm_matrix(n):
    np.random.seed(41)
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    return A
def generate_psd_matrix(n):
    np.random.seed(42)
    n = n //2 # (2*n, 2*n) -> (n, n)
    cov = random_covariance(n)
    O = Xmat(n) @ Amat(cov)
    return O.real

def _haf_walrus(mat, n):
    return hafnian(mat.reshape(n, n))
def _haf_walrus_loop(mat, n):
    return loop_hafnian(mat.reshape(n, n))
def _tor_walrus(mat, n):
    return tor(mat.reshape(n, n))
def _tor_walrus_loop(mat, n):
    gamma = mat.reshape(n, n).diagonal()
    return ltor(mat.reshape(n, n), gamma)

def benchmark_haf_tor(func, rand_func, mat_size, repeat, loop):
    T_wal = [ ]
    for k in range(repeat):
        t = []
        for i in list(range(2, mat_size+1))[::2]:
            a = rand_func(i)
            batch_a1 = np.array([a.flatten()] *  batch)
            t1 = time.time()
            wal_haf_tor = np.vectorize(func, signature='(n),()-> ()' )(batch_a1, i)
            t2 = time.time()
            print('thewalrus', 'repeat:', k, 'size:', int(i), end='\r')
            t.append(t2-t1)
        T_wal.append(t)
    print('thewalrus', np.array(T_wal))
    if haf_size is not None:
        s = 'haf'
    if tor_size is not None:
        s = 'tor'
    np.save(f'thewalrus_batch_{batch}_size_{mat_size}_loop_{loop}_' + s + '.npy', np.array(T_wal))
    return T_wal



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='thewalrus')
    parser.add_argument('--haf_size', type=int, default=None)
    parser.add_argument('--haf_loop', action='store_true', default=False)
    parser.add_argument('--tor_loop', action='store_true', default=False)
    parser.add_argument('--tor_size', type=int, default=None)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    haf_size = args.haf_size
    tor_size = args.tor_size
    repeat = args.repeat
    batch = args.batch
    haf_loop = args.haf_loop
    tor_loop = args.tor_loop

    print('the matrix size for hafian is 2 - ', haf_size, ', loop =', haf_loop)
    print('the matrix size for torontonian is 2 - ', tor_size, ', loop =', tor_loop)
    print('the batch is ', batch)
    print('the repeat times is ', repeat)

    if haf_size is not None:
        assert haf_size%2 == 0
        if haf_loop:
            func = _haf_walrus_loop
        else:
            func = _haf_walrus
        rand_func = generate_symm_matrix
        mat_size = haf_size
        print("run hafnian calcluation......")
        T_wal = benchmark_haf_tor(func, rand_func, mat_size, repeat, haf_loop)

    if tor_size is not None:
        assert tor_size%2 == 0
        if tor_loop:
            func = _tor_walrus_loop
        else:
            func = _tor_walrus
        rand_func = generate_psd_matrix
        mat_size = tor_size
        print("run torontonian calcluation......")
        T_wal = benchmark_haf_tor(func, rand_func, mat_size, repeat, tor_loop)
