import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename')
    parser.add_argument('-p', '--reps', type=int, default=4)
    parser.add_argument('-m', '--memory', type=int, default=4000)
    parser.add_argument('-n', '--shots', type=int, default=2000)
    parser.add_argument('-i', '--maxiter', type=int, default=100)
    parser.add_argument('-l', '--lamda', type=float, default=5)
    parser.add_argument('-b', '--blocking', type=float, default=30)
    parser.add_argument('--hardware', action='store_true', default=True)
    parser.add_argument('--noisy', action='store_true', default=False)
    parser.add_argument('--init', choices=['ramp', 'random'], default='random')
    parser.add_argument('-M', '--method', choices=['Nelder-Mead', 'Powell', 'COBYLA', 'spsa', 'none'], default='Powell')
    return parser