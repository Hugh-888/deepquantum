import strawberryfields as sf
import time
import argparse
import numpy as np

from scipy.stats import unitary_group

# Using Strawberryfields gaussian backend to perform Gaussian Boson Sampling.
# This script performs Gaussian Boson Sampling with specified shot counts and repeat times.
# It measures the time taken for each sampling process.

# Example usage: python gbs_sf_benchmark.py --modes_list 2 3 4 --shots 200 --repeat 2
# - modes_list: A list of modes for constructing GBS circuit (e.g., 2, 3, 4, etc.)
# - shots: Shots for GBS sampling. Default: 200
# - repeat: Number of times to repeat the sampling process. Default: 1
# The time taken for each sampling is recorded and saved in ".npy" file.

def benchmark_gbs(modes_list, shots, repeat):
    T_sf = [ ]
    r = np.arcsinh(1)
    hbar = 2
    for k in range(repeat):
        t_sf = []
        for nmode in modes_list:
            nmode = int(nmode)
            gbs = sf.Program(nmode)
            np.random.seed(41)
            U = unitary_group.rvs(nmode)
            with gbs.context as q:
                for i in range(nmode):
                    sf.ops.Sgate(r) | q[i]
                sf.ops.Interferometer(U) | tuple(q[i] for i in range(nmode))
                sf.ops.MeasureFock() | tuple(q[i] for i in range(nmode))
            eng = sf.Engine("gaussian", backend_options={"hbar": hbar}) # cutoff=5 by default in sf
            t1 = time.time()
            results = eng.run(gbs, shots=shots)
            t2 = time.time()
            print('sf', 'repeat:', k, 'nmodes:', nmode, end='\r')
            t_sf.append(t2-t1)
        T_sf.append(t_sf)
    np.save(f"sf_gbs_shots{shots}_nmodes{nmode}.npy", np.array(T_sf)) # (repeat, len(modes_list))
    print('SF', np.array(T_sf))
    return T_sf


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

    T_sf = benchmark_gbs(modes_list, shots, repeat)