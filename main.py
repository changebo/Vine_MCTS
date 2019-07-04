from vinemcts.mcts import *
import numpy as np
import os
import random
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", action="store", dest="file_prefix", default="", help="File prefix"
)
parser.add_argument(
    "-nw",
    action="store",
    dest="num_worker",
    default=1,
    type=int,
    help="Number of workers to run in parallel (default: 1)",
)
parser.add_argument(
    "-fpu",
    action="store",
    dest="FPU",
    default=1.0,
    type=float,
    help="First play urgency (default: 1.0)",
)
parser.add_argument(
    "-pb",
    action="store",
    dest="PB",
    default=0.1,
    type=float,
    help="Progressive bias (default: 0.1)",
)
parser.add_argument(
    "-log_freq",
    action="store",
    dest="log_freq",
    default=100,
    type=int,
    help="Log Frequency (default: 100)",
)
parser.add_argument(
    "-ntrunc",
    action="store",
    dest="ntrunc",
    default="",
    type=str,
    help='A list of truncation level. For example, "2,3,4". By default, levels from 1 to d-1.',
)
parser.add_argument(
    "-seed", action="store", dest="seed", default=1, type=int, help="Seed (default: 1)"
)
parser.add_argument(
    "-max_iter",
    action="store",
    dest="max_iter",
    default=5000,
    type=int,
    help="Maximum number of iterations (default: 5000)",
)
args = parser.parse_args()


def func(file):
    f = open(os.path.join("data/corr", file), "r")
    n_sample = int(f.readline())
    f.close()

    rmat = np.loadtxt(os.path.join("data/corr", file), delimiter=",", skiprows=1)
    d = len(rmat)
    if args.ntrunc == "":
        ntruncs = range(1, d)
    else:
        ntruncs = [int(ell) for ell in args.ntrunc.split(",")]

    for ntruc in ntruncs:
        assert ntruc >= 1 and ntruc < d

        print("Seed: " + str(args.seed))

        random.seed(args.seed)
        np.random.seed(args.seed)

        output_dir = os.path.join(
            "output",
            file.split(".")[0]
            + "_trunc_"
            + str(ntruc)
            + "_seed_"
            + str(args.seed)
            + ".txt",
        )
        best_vine = mcts_vine(
            rmat,
            n_sample=n_sample,
            output_dir=output_dir,
            ntrunc=ntruc,
            itermax=args.max_iter,
            FPU=args.FPU,
            PB=args.PB,
            log_freq=args.log_freq,
        )

        best_vine_array = best_vine.to_vine_array()
        best_vine_array = best_vine_array.flatten()
        best_vine_array = np.array2string(best_vine_array, separator=",")
        print(best_vine_array)


if __name__ == "__main__":
    print("Number of workers: %d" % args.num_worker)
    pool = multiprocessing.Pool(args.num_worker)
    files = [f for f in os.listdir("data/corr") if f.startswith(args.file_prefix)]
    pool.map(func, files)
