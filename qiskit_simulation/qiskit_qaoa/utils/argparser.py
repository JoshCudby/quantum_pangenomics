import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename')
    parser.add_argument('-p', '--reps', type=int, default=4)
    parser.add_argument('-m', '--memory', type=int, default=4000)
    parser.add_argument('-M', '--method', type=str, default='')
    parser.add_argument('-n', '--shots', type=int, default=2000)
    parser.add_argument('-a', '--alpha', type=float, default=0.25)
    parser.add_argument('--hardware', action='store_true', default=False)
    parser.add_argument('--noisy', action='store_true', default=False)
    parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
    return parser