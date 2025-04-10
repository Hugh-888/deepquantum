import deepquantum as dq
import strawberryfields as sf
import time
import argparse
import torch
import numpy as np

# Using Strawberryfields Fock backend to sample the quadratures of fock state.
# This script performs fock state homodyne sampling with specified shot counts and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python fock_homodyne_sample_sf_benchmark.py --shots_list 1e2 5e2 --batch 1 --repeat 2

# - shots_list: A list of shot counts for sampling (e.g., 100 500 1000, etc.)
# - batch: for loop times since multiple shots not supported with batch data in strawberryfields
#  Default: 1
# - repeat: Number of times to repeat the sampling process. Default: 1
# The time taken for each sampling is recorded and saved in ".npy" file.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fock homodyne sampling')
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
            t_sf = []
            for i in shots_list:
                prog_fock = sf.Program(3)
                hbar = 2
                with prog_fock.context as q:
                    sf.ops.Fock(1) | q[0]
                    sf.ops.Fock(1) | q[1]
                    sf.ops.Fock(1) | q[2]

                    sf.ops.BSgate(np.pi/3, np.pi/3) | (q[0], q[1])
                    sf.ops.BSgate(np.pi/3, np.pi/3) | (q[1], q[2])
                    sf.ops.MeasureX | q[1]
                eng = sf.Engine("fock", backend_options={"hbar": hbar, 'cutoff_dim':4}) # xpxp  order
                t1 = time.time()
                for __ in range(int(i)):
                    sample_x = eng.run(prog_fock, shots=1).samples # multiple shots not supported in this case in sf
                t2 = time.time()
                print('sf', 'repeat:', k, 'shots:', int(i), 'batch:', _ , end='\r')
                t_sf.append(t2-t1)
            T_sf.append(t_sf)
        T_sf_batch.append(torch.tensor(T_sf))

    np.save("sf_batch_%d_fock_homodyne.npy"%batch, torch.stack(T_sf_batch)) # (batch, repeat, len(shots_list))
    print('SF', torch.stack(T_sf_batch))




