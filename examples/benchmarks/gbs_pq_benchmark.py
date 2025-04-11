import piquasso as pq
import time
import argparse
import numpy as np

from scipy.stats import unitary_group

# Using Piquasso gaussian backend to perform Gaussian Boson Sampling.
# This script performs Gaussian Boson Sampling with specified shot counts and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python gbs_pq_benchmark.py --modes_list 2 3 4 --shots 200 --repeat 2
# - modes_list: A list of modes for constructing GBS circuit (e.g., 2, 3, 4, etc.)
# - shots: Shots for GBS sampling. Default: 200
# - repeat: Number of times to repeat the sampling process. Default: 1
# The time taken for each sampling is recorded and saved in ".npy" file.

def benchmark_gbs(modes_list, shots, repeat):
    T_pq = [ ]
    r = np.arcsinh(1)
    for k in range(repeat):
        t_pq = []
        for nmode in modes_list:
            nmode = int(nmode)
            np.random.seed(41)
            U = unitary_group.rvs(nmode)
            with pq.Program() as program:
                for i in range(nmode):
                    pq.Q(i) | pq.Squeezing(r=r)
                pq.Q(all) | pq.Interferometer(U)
                pq.Q(all) | pq.ParticleNumberMeasurement()
            simulator = pq.GaussianSimulator(d=nmode, config=pq.Config(measurement_cutoff=5 + 1))
            t1 = time.time()
            results = simulator.execute(program, shots=shots) # cutoff = 5 by default
            t2 = time.time()
            print('pq', 'repeat:', k, 'nmodes:', nmode, end='\r')
            t_pq.append(t2-t1)
        T_pq.append(t_pq)
    np.save(f"pq_gbs_shots{shots}_nmodes{nmode}.npy", np.array(T_pq)) # (repeat, len(modes_list))
    print('PQ', np.array(T_pq))
    return T_pq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GBS sampling')
    parser.add_argument('--modes_list', type=float, nargs='+',
                    help='List of shots, e.g., 2 3 4 5 6, etc')
    parser.add_argument('--shots', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    shots = args.shots
    modes_list = args.modes_list
    repeat = args.repeat
    print('the modes list is ', modes_list)
    print('the sampling shots is ', shots)
    print('the repeat times is ', repeat)

    T_pq = benchmark_gbs(modes_list, shots, repeat)
