
import argparse
import numpy as np
import time
import torch
import deepquantum.photonic as dqp

from thewalrus.random import random_covariance
from thewalrus.quantum.conversions import Amat, Xmat

# Using DeepQuantum to calculate the hafnian and torontonian of random matrix.
# This script performs hafnian and torontonian calculation with specified matrix size, batch sizes, and repeat times.
# It measures the time taken for each process.

# Example usage:
# run in cpu
# python haf_tor_deepquantum_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2
# python haf_tor_deepquantum_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2

# run in gpu
# python haf_tor_deepquantum_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2 --device cuda
# python haf_tor_deepquantum_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2 --device cuda

#run loop hafnian
# python haf_tor_deepquantum_batch_benchmark.py --haf_size 10 --batch 2 --repeat 2 --haf_loop

#run loop torontonian
# python haf_tor_deepquantum_batch_benchmark.py --tor_size 10 --batch 2 --repeat 2 --tor_loop

# - haf_size: the max size of matrix for hafnian (2, 4, ..., haf_size). Default: ``None``
# - haf_loop: Whether to calculate the loop hafnian. Default: ``False``
# - tor_loop: Whether to calculate the loop torontonian. Default: ``False``
# - tor_size: the max size of matrix for torontonian (2, 4, ..., tor_size). Default: ``None``
# - batch: Number of matrices to process for each calculation. Default: 1
# - repeat: Number of times to repeat the process. Default: 1
# - device: 'cpu' or 'cuda'. Default: ``cpu``
# The time taken for each process is recorded and saved in ".npy" file.

def generate_symm_matrix(n):
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    return A
def generate_psd_matrix(n):
    cov = random_covariance(n)
    O = Xmat(n) @ Amat(cov)
    return O.real # (2*n, 2*n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='thewalrus')
    parser.add_argument('--haf_size', type=int, default=None)
    parser.add_argument('--haf_loop', action='store_true', default=False)
    parser.add_argument('--tor_loop', action='store_true', default=False)
    parser.add_argument('--tor_size', type=int, default=None)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    haf_size = args.haf_size
    tor_size = args.tor_size
    repeat = args.repeat
    batch = args.batch
    device = args.device
    haf_loop = args.haf_loop
    tor_loop = args.tor_loop
    print('the matrix size for hafian is 2 - ', haf_size, ', loop =', haf_loop)
    print('the matrix size for torontonian is 2 - ', tor_size, ', loop =', tor_loop)
    print('the batch is ', batch)
    print('the repeat times is ', repeat)
    print('the device is ', device)

    # hafnian
    if haf_size is not None:
        print("run hafnian calcluation......")
        assert haf_size%2 == 0
        T_dq = [ ]
        for k in range(repeat):
            t = []
            for i in list(range(2, haf_size+1))[::2]:
                a = generate_symm_matrix(i)
                batch_a1 = torch.tensor(np.array([a] * batch), device=device)
                t1 = time.time()
                dq_haf = torch.vmap(dqp.hafnian, in_dims=(0, None))(batch_a1, haf_loop)
                t2 = time.time()
                print('deepquantum', 'repeat:', k, 'size:', int(i), end='\r')
                t.append(t2-t1)
            T_dq.append(t)
        print('deepquantum', torch.tensor(T_dq))
        np.save(f"dq_batch_{batch}_size_{haf_size}_loop_{haf_loop}_{device}_haf.npy", torch.tensor(T_dq))


    # torontonian
    if tor_size is not None:
        print("run torontonian calcluation......")
        assert tor_size%2 == 0
        T_dq = [ ]
        for k in range(repeat):
            t = []
            for i in list(range(2, tor_size+1))[::2]:
                a = generate_psd_matrix(i//2)
                batch_a1 = torch.tensor(np.array([a] * batch), device=device)
                gamma = torch.diagonal(batch_a1, dim1=1, dim2=2)
                t1 = time.time()
                dq_tor = torch.vmap(dqp.torontonian, in_dims=(0, 0))(batch_a1, gamma)
                t2 = time.time()
                print('deepquantum', 'repeat:', k, 'size:', int(i), end='\r')
                t.append(t2-t1)
            T_dq.append(t)
        print('deepquantum', torch.tensor(T_dq))
        np.save(f"dq_batch_{batch}_size_{tor_size}__loop_{tor_loop}_{device}_tor.npy", torch.tensor(T_dq))




