import deepquantum as dq
import strawberryfields as sf
import time
import argparse
import torch
import numpy as np

# Using Strawberryfields bosonic backend to sample the quadratures of GKP state.
# This script performs GKP state sampling with specified shot counts and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python gkp_sample_sf_benchmark.py --shots_list 1e2 5e2 --batch 2 --repeat 2
# - shots_list: A list of shot counts for sampling (e.g., 100, 500, 1000, etc.)
# - batch: for loop times since multiple shots not supported with batch data in strawberryfields
#  Default: 1
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
    batch = args.batch
    repeat = args.repeat
    print('the sample shots is ', shots_list)
    print('the batch is ', batch)
    print('the repeat times is ', repeat)

    if shots_list is None:
        shots_list = [1e2, 5e2, 1e3, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4]
    T_sf_batch = [ ]
    for _ in range(batch):
        T_sf = [ ]
        for k in range(repeat):
            nmodes = 1
            t_sf = []
            for i in shots_list:
                prog_bosonic = sf.Program(nmodes)
                hbar = 2
                with prog_bosonic.context as q:
                    sf.ops.GKP(state=[0, 0], ampl_cutoff=0.01, epsilon=0.05) | q[0] # superposition of 4 states
                    sf.ops.MeasureX | q[0]
                eng = sf.Engine("bosonic", backend_options={"hbar": hbar}) # xpxp  order
                t1 = time.time()
                sample_x = eng.run(prog_bosonic, shots=int(i)).samples[:,0] # multiple shots not supported with batch data in sf
                t2 = time.time()
                print('sf', 'repeat:', k, 'shots:', int(i), 'batch:', _ , end='\r')
                t_sf.append(t2-t1)
            T_sf.append(t_sf)
        T_sf_batch.append(torch.tensor(T_sf))

    np.save("sf_batch_%d_gkp.npy"%batch, torch.stack(T_sf_batch)) # (batch, repeat, len(shots_list))
    print('SF', torch.stack(T_sf_batch))