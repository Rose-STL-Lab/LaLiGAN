import numpy as np
import argparse


def triangle(n, w):
    x1 = np.linspace(start=0, stop=1, num=w // 2, endpoint=False)
    x2 = np.linspace(start=1, stop=0, num=w - w // 2, endpoint=False)
    x = np.concatenate([x1, x2, np.zeros(n - w)])
    assert len(x) == n
    return x


def rectangle(n, w):
    x = np.ones(w)
    x = np.concatenate([x, np.zeros(n - w)])
    x = x.astype(float)
    assert len(x) == n
    return x


def half_circle(n, w):
    x = np.sin(np.linspace(start=0, stop=np.pi, num=w, endpoint=False))
    x = np.concatenate([x, np.zeros(n - w)])
    assert len(x) == n
    return x


def cyclic_shift(x, d):
    x = np.concatenate([x[d:], x[:d]])
    return x


def random_cyclic_shift(x):
    n = len(x)
    d = np.random.randint(n)
    x = cyclic_shift(x, d)
    return x, d / n


def get_double_bump_data(n_samples, signal_len=64, bump_len=16, fix_d1=-1, fix_d2=-1):
    x_list = np.empty((n_samples, signal_len))
    d_list = np.empty((n_samples, 2))
    for i in range(n_samples):
        if fix_d1 >= 0:
            sig1 = cyclic_shift(rectangle(signal_len, bump_len), fix_d1 % signal_len)
            d1 = fix_d1 % signal_len
        else:
            sig1, d1 = random_cyclic_shift(rectangle(signal_len, bump_len))
        if fix_d2 >= 0:
            sig2 = cyclic_shift(triangle(signal_len, bump_len), fix_d2 % signal_len)
            d2 = fix_d2 % signal_len
        else:
            sig2, d2 = random_cyclic_shift(triangle(signal_len, bump_len))
        x_list[i] = 0.5 * (sig1 + sig2)
        d_list[i] = np.array([d1, d2])

    return x_list, d_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--signal_len', type=int, default=64)
    parser.add_argument('--bump_len', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='/data/LaSSI/doublebump')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--fix_d1', type=int, default=-1)
    parser.add_argument('--fix_d2', type=int, default=-1)
    args = parser.parse_args()
    x_list, d_list = get_double_bump_data(args.n_samples, args.signal_len, args.bump_len, args.fix_d1, args.fix_d2)
    np.save(f'{args.data_dir}/{args.name}_x.npy', x_list)
    np.save(f'{args.data_dir}/{args.name}_d.npy', d_list)
