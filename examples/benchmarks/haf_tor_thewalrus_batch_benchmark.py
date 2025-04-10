
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
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    return A
def generate_psd_matrix(n):
    cov = random_covariance(n)
    O = Xmat(n) @ Amat(cov)
    return O.real # (2*n, 2*n)

def _haf_walrus(mat, n):
    return hafnian(mat.reshape(n, n))
def _haf_walrus_loop(mat, n):
    return loop_hafnian(mat.reshape(n, n))
def _tor_walrus(mat, n):
    return tor(mat.reshape(n, n))
def _tor_walrus_loop(mat, n):
    gamma = mat.reshape(n, n).diagonal()
    return ltor(mat.reshape(n, n), gamma)

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

    # hafnian
    if haf_size is not None:
        if haf_loop:
            haf_func = _haf_walrus_loop
        else:
            haf_func = _haf_walrus
        print("run hafnian calcluation......")
        assert haf_size%2 == 0
        T_wal = [ ]
        for k in range(repeat):
            t = []
            for i in list(range(2, haf_size+1))[::2]:
                a = generate_symm_matrix(i)
                batch_a1 = np.array([a.flatten()] *  batch)
                t1 = time.time()
                walrus_haf = np.vectorize(haf_func, signature='(n),()-> ()' )(batch_a1, i)
                t2 = time.time()
                print('thewalrus', 'repeat:', k, 'size:', int(i), end='\r')
                t.append(t2-t1)
            T_wal.append(t)
        print('thewalrus', torch.tensor(T_wal))
        np.save(f"thewalrus_batch_{batch}_size_{haf_size}_loop_{haf_loop}_haf.npy", torch.tensor(T_wal))


    #  torontonian
    if tor_size is not None:
        if tor_loop:
            tor_func = _tor_walrus_loop
        else:
            tor_func = _tor_walrus
        print("run torontonian calcluation......")
        assert tor_size%2 == 0
        T_wal = [ ]
        for k in range(repeat):
            t = []
            for i in list(range(2, tor_size+1))[::2]:
                a = generate_psd_matrix(i//2)
                batch_a1 = np.array([a.flatten()] *  batch)
                t1 = time.time()
                walrus_tor = np.vectorize(tor_func, signature='(n),()-> ()' )(batch_a1, i)
                t2 = time.time()
                print('thewalrus', 'repeat:', k, 'size:', int(i), end='\r')
                t.append(t2-t1)
            T_wal.append(t)
        print('thewalrus', torch.tensor(T_wal))
        np.save(f"thewalrus_batch_{batch}_size_{tor_size}_loop_{tor_loop}_tor.npy", torch.tensor(T_wal))