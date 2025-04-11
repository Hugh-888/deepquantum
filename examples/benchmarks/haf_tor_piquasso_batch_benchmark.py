
import argparse
import numpy as np
import time

from piquasso._math.hafnian import hafnian_with_reduction
from piquasso._math.hafnian import loop_hafnian_with_reduction
from piquasso._math.torontonian import torontonian, loop_torontonian
from thewalrus.random import random_covariance
from thewalrus.quantum.conversions import Amat, Xmat

# Using Piquasso to calculate the hafnian and torontonian of random matrix.
# This script performs hafnian and torontonian calculation with specified matrix size, batch sizes, and repeat times.
# It measures the time taken for each process.

# Example usage:
# plain hafnian
# python haf_tor_piquasso_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2
# loop hafnian
# python haf_tor_piquasso_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2 --haf_loop
# torontonian
# python haf_tor_piquasso_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2
# loop torontonian
# python haf_tor_piquasso_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2 --tor_loop


# - haf_size: the max size of matrix for hafnian (2, 4, ..., haf_size). Default: ``None``
# - haf_loop: Whether to calculate the loop hafnian. Default: ``False``
# - tor_loop: Whether to calculate the torontonian hafnian. Default: ``False``
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

def _haf_pq(mat, n):
    return hafnian_with_reduction(mat.reshape(n, n), np.array([1]*n))
def _haf_pq_loop(mat, n):
    diag = mat.reshape(n, n).diagonal()
    return loop_hafnian_with_reduction(mat.reshape(n, n), diag, np.array([1]*n))
def _tor_pq(mat, n):
    return torontonian(mat.reshape(n, n))
def _tor_pq_loop(mat, n):
    gamma = mat.reshape(n, n).diagonal()
    return loop_torontonian(mat.reshape(n, n), gamma)

def benchmark_haf_tor(func, rand_func, mat_size, repeat, loop):
    T_pq = [ ]
    for k in range(repeat):
        t = []
        for i in list(range(2, mat_size+1))[::2]:
            a = rand_func(i)
            batch_a1 = np.array([a.flatten()] *  batch)
            t1 = time.time()
            pq_haf_tor = np.vectorize(func, signature='(n),()-> ()' )(batch_a1, i)
            t2 = time.time()
            print('piquasso', 'repeat:', k, 'size:', int(i), end='\r')
            t.append(t2-t1)
        T_pq.append(t)
    print('piquasso', np.array(T_pq))
    if haf_size is not None:
        s = 'haf'
    if tor_size is not None:
        s = 'tor'
    np.save(f'pq_batch_{batch}_size_{mat_size}_loop_{loop}_' + s + '.npy', np.array(T_pq))
    return T_pq

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
            func = _haf_pq_loop
        else:
            func = _haf_pq
        rand_func = generate_symm_matrix
        mat_size = haf_size
        print("run hafnian calcluation......")
        T_pq = benchmark_haf_tor(func, rand_func, mat_size, repeat, haf_loop)

    if tor_size is not None:
        assert tor_size%2 == 0
        if tor_loop:
            func = _tor_pq_loop
        else:
            func = _tor_pq
        rand_func = generate_psd_matrix
        mat_size = tor_size
        print("run torontonian calcluation......")
        T_pq = benchmark_haf_tor(func, rand_func, mat_size, repeat, tor_loop)
