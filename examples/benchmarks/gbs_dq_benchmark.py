import deepquantum as dq
import time
import argparse
import torch
import numpy as np

from scipy.stats import unitary_group

# Using Deepquantum gaussian backend to perform Gaussian Boson Sampling.
# This script performs Gaussian Boson Sampling with specified shot counts and repeat times.
# It measures the time taken for each sampling process.

# Example usage:
# run in cpu
# python gbs_dq_benchmark.py --modes_list 2 3 4 --shots 200 --repeat 2
# run in gpu
# python gbs_dq_benchmark.py --modes_list 2 3 4 --shots 200 --repeat 2 --device cuda

# - modes_list: A list of modes for constructing GBS circuit (e.g., 2, 3, 4, etc.)
# - shots: Shots for GBS sampling. Default: 200
# - repeat: Number of times to repeat the sampling process. Default: 1
# - device: 'cpu' or 'cuda'. Default: ``cpu``
# The time taken for each sampling is recorded and saved in ".npy" file.

def benchmark_gbs(modes_list, shots, repeat, device):
    T_dq = [ ]
    r = np.arcsinh(1)
    for k in range(repeat):
        t_dq = []
        for nmode in modes_list:
            nmode = int(nmode)
            np.random.seed(41)
            U = unitary_group.rvs(nmode)
            cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', backend='gaussian', cutoff=6)
            for i in range(nmode):
                cir.s(i, r=r)
            cir.any(U)
            cir.to(device)
            t1 = time.time()
            cir()
            samples_dq = cir.measure(shots=shots)
            t2 = time.time()
            print('dq', 'repeat:', k, 'nmodes:', nmode, end='\r')
            t_dq.append(t2-t1)
        T_dq.append(t_dq)
    np.save(f"dq_gbs_shots{shots}_nmodes{nmode}_{device}.npy", np.array(T_dq)) # (repeat, len(modes_list))
    print('DQ', np.array(T_dq))
    return T_dq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GBS sampling')
    parser.add_argument('--modes_list', type=float, nargs='+',
                    help='List of shots, e.g., 2 3 4 5 6, etc')
    parser.add_argument('--shots', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    shots = args.shots
    modes_list = args.modes_list
    repeat = args.repeat
    device = args.device
    print('the modes list is ', modes_list)
    print('the sampling shots is ', shots)
    print('the repeat times is ', repeat)
    print('the device is ', device)

    T_dq = benchmark_gbs(modes_list, shots, repeat, device)
    