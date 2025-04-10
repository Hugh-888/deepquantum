import deepquantum as dq
import strawberryfields as sf
import time
import argparse
import torch
import numpy as np

# Using DeepQuantum bosonic backend to sample the quadratures of GKP state.
# This script performs GKP state sampling with specified shot counts, batch sizes, and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python gkp_sample_dq_batch_benchmark.py --shots_list 1e2 5e2 --batch 2 --repeat 10
# - shots_list: A list of shot counts for sampling (e.g., 100, 500, 1000, etc.)
# - batch: Number of samples to process in each batch. Default: 1
# - repeat: Number of times to repeat the sampling process. Default: 1
# The time taken for each sampling is recorded and saved in ".npy" file.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GKP sampling test')
    parser.add_argument('--shots_list', type=float, nargs='+',
                    help='List of shots, e.g., 100 500 1000 5000 10000')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    shots_list = args.shots_list
    repeat = args.repeat
    batch = args.batch
    print('the sample shots is ', shots_list)
    print('the encode data batch is ', batch)
    print('the repeat times is ', repeat)

    if shots_list is None:
        shots_list = [1e2, 5e2, 1e3, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4]
    T_dq = [ ]
    for k in range(repeat):
        t = []
        for i in shots_list:
            cir = dq.QumodeCircuit(nmode=1, init_state='vac', cutoff=3, backend='bosonic')
            cir.gkp(0, theta=0., phi=0., amp_cutoff=0.01, epsilon=0.05)
            cir.s(0, 0., encode=True)
            cir.homodyne(0, 0)
            data = torch.tensor([[0, 0]]*batch)
            test = cir(data=data)
            t1 = time.time()
            sample_re = cir.measure_homodyne(shots=int(i))
            t2 = time.time()
            print('dq', 'repeat:', k, 'shots:', int(i), end='\r')
            t.append(t2-t1)
        T_dq.append(t)

    np.save("dq_batch_%d_gkp.npy"%batch, torch.tensor(T_dq))
    print('DQ', torch.tensor(T_dq))




