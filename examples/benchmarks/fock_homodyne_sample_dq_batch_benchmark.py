import deepquantum as dq
import time
import argparse
import torch
import numpy as np

# Using DeepQuantum Fock backend to sample the quadratures of fock state.
# This script performs fock state homodyne sampling with specified shot counts, batch sizes, and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python fock_homodyne_sample_dq_batch_benchmark.py --shots_list 1e3 5e3 --batch 1 --repeat 2
# - shots_list: A list of shot counts for sampling (e.g., 100 500 1000, etc.)
# - batch: Number of encodig data to process in each sampling process. Default: 1
# - repeat: Number of times to repeat the sampling process. Default: 1
# - den_mat: Whether to use density matrix representation. Default: ``False``
# The time taken for each sampling is recorded and saved in ".npy" file.

def benchmark_fock_homodyne(shots_list, repeat, batch, den_mat):
    T_dq = [ ]
    for k in range(repeat):
        t = []
        for i in shots_list:
            cir = dq.QumodeCircuit(nmode=3, init_state=[(1,[1, 1, 1])], cutoff=4, backend='fock', basis=False, den_mat=den_mat)
            cir.bs([0,1], inputs=[np.pi/3, np.pi/3])
            cir.bs([1,2], inputs=[np.pi/3, np.pi/3])
            data = torch.tensor([[0, 0]] * batch)
            t1 = time.time()
            cir(data=data)
            samples = cir.measure_homodyne(wires=[1], shots=int(i))
            t2 = time.time()
            print('dq', 'repeat:', k, 'shots:', int(i), end='\r')
            t.append(t2-t1)
        T_dq.append(t)
    np.save("dq_batch_%d_fock_homodyne.npy"%batch, torch.tensor(T_dq))
    print('DQ', torch.tensor(T_dq))
    return T_dq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fock homodyne sampling')
    parser.add_argument('--shots_list', type=float, nargs='+',
                        help='List of shots, e.g., 100 500 1000 5000 10000',
                        default=None)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--den_mat', type=bool, default=False)
    args = parser.parse_args()
    shots_list = args.shots_list
    repeat = args.repeat
    batch = args.batch
    den_mat = args.den_mat
    if shots_list is None:
        shots_list = [1e2, 5e2, 1e3, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4]
    print('the sample shots is ', shots_list)
    print('the batch is ', batch)
    print('the repeat times is ', repeat)
    print('the den_mat is ', den_mat)
    T_dq = benchmark_fock_homodyne(shots_list, repeat, batch, den_mat)




