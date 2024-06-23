import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from data_utils.pendulum import get_pendulum_data, get_low_dim_pendulum_data
from data_utils.lotka import get_lv_data
from data_utils.double_bump import get_double_bump_data

data_path = '../data'


# Modified from SINDy AE: https://github.com/kpchamp/SindyAutoencoders/blob/master/examples/rd/example_reactiondiffusion.py
class ReactionDiffusionDataset(Dataset):
    def __init__(self, path=f'{data_path}/reaction_diffusion.mat', random=False, mode='train', downsample=False):
        data = sio.loadmat(path)
        n_samples = data['t'].size
        n = data['x'].size
        N = n*n

        data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
        data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

        if not random:
            # consecutive samples
            training_samples = np.arange(int(.8*n_samples))
            val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
            test_samples = np.arange(int(.9*n_samples), n_samples)
        else:
            # random samples
            perm = np.random.permutation(int(.9*n_samples))
            training_samples = perm[:int(.8*n_samples)]
            val_samples = perm[int(.8*n_samples):]
            test_samples = np.arange(int(.9*n_samples), n_samples)

        if mode == 'train':
            samples = training_samples
        elif mode == 'val':
            samples = val_samples
        elif mode == 'test':
            samples = test_samples

        self.data = {
            't': data['t'][samples],
            'y1': data['x'].T,
            'y2': data['y'].T,
            'x': data['uf'][:, :, samples].reshape((N, -1)).T,
            'dx': data['duf'][:, :, samples].reshape((N, -1)).T
            }
        if downsample:
            # reshape each x (10000) to (100, 100) and downsample to (28, 28)
            from scipy.ndimage import zoom
            x = self.data['x'].reshape((-1, 100, 100))
            dx = self.data['dx'].reshape((-1, 100, 100))
            downsampled_x = np.zeros((self.data['x'].shape[0], 28, 28))
            downsampled_dx = np.zeros((self.data['x'].shape[0], 28, 28))
            for i in range(self.data['x'].shape[0]):
                downsampled_x[i] = zoom(x[i], 0.28)
                downsampled_dx[i] = zoom(dx[i], 0.28)
            self.data['x'] = downsampled_x.reshape((-1, 28*28))
            self.data['dx'] = downsampled_dx.reshape((-1, 28*28))

        self.data = {k: torch.FloatTensor(v) for k, v in self.data.items()}

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, idx):
        return self.data['x'][idx], self.data['dx'][idx], self.data['dx'][idx]


class MultiTimestepReactionDiffusionDataset(Dataset):
    def __init__(self, path=f'{data_path}/reaction_diffusion.mat', n_timesteps=2, mode='train', downsample=False):
        data = sio.loadmat(path)
        n_samples = data['t'].size

        data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
        data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

        training_samples = np.arange(int(.8*n_samples))
        val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
        test_samples = np.arange(int(.9*n_samples), n_samples)
        if mode == 'train':
            samples = training_samples
        elif mode == 'val':
            samples = val_samples
        elif mode == 'test':
            samples = test_samples

        if downsample:
            # reshape each x (10000) to (100, 100) and downsample to (28, 28)
            from scipy.ndimage import zoom
            x = data['uf'].reshape(100, 100, -1)
            dx = data['duf'].reshape(100, 100, -1)
            downsampled_x = np.zeros((28, 28, x.shape[-1]))
            downsampled_dx = np.zeros((28, 28, x.shape[-1]))
            for i in range(x.shape[-1]):
                downsampled_x[..., i] = zoom(x[..., i], 0.28)
                downsampled_dx[..., i] = zoom(dx[..., i], 0.28)
            data['uf'] = downsampled_x
            data['duf'] = downsampled_dx

        self.data = []
        for i in range(n_timesteps, len(samples)):
            self.data.append({
                'x': torch.FloatTensor(np.transpose(data['uf'][:, :, samples[i-n_timesteps:i]], axes=(2, 0, 1)).reshape((n_timesteps, -1))),
                'dx': torch.FloatTensor(np.transpose(data['duf'][:, :, samples[i-n_timesteps:i]], axes=(2, 0, 1)).reshape((n_timesteps, -1)))
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['dx'], self.data[idx]['dx']


class RotatingShelfDataset(Dataset):
    def __init__(self, path=f'{data_path}/rotobj', mode='train', flatten=False):
        self.data = np.load(f'{path}/{mode}.npy')
        self.data = self.data.astype(np.float32)
        self.data = self.data / 255.
        self.data = torch.FloatTensor(self.data)
        if flatten:
            self.data = self.data.reshape((self.data.shape[0], -1))
        self.rotations = np.load(f'{path}/{mode}_rot.npy')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx], self.rotations[idx]


class PendulumDataset(Dataset):
    def __init__(self, path=f'{data_path}/pendulum', n_timesteps=2, mode='train'):
        super().__init__()
        try:
            print(f'Loading existing pendulum {mode} data...')
            x = torch.load(f'{path}/{mode}-x.pt')
            dx = torch.load(f'{path}/{mode}-dx.pt')
            ddx = torch.load(f'{path}/{mode}-ddx.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating pendulum {mode} data...')
            n_ics = 200 if mode == 'train' else 20
            data = get_pendulum_data(n_ics=n_ics)
            x = data['x'].reshape(n_ics, -1, data['x'].shape[-1])
            dx = data['dx'].reshape(n_ics, -1, data['dx'].shape[-1])
            ddx = data['ddx'].reshape(n_ics, -1, data['ddx'].shape[-1])
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            ddx = torch.FloatTensor(ddx)
            torch.save(x, f'{path}/{mode}-x.pt')
            torch.save(dx, f'{path}/{mode}-dx.pt')
            torch.save(ddx, f'{path}/{mode}-ddx.pt')
        self.n_timesteps = n_timesteps
        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = []
        self.dx = []
        self.ddx = []
        for i in range(n_ics):
            for j in range(n_steps-n_timesteps):
                self.x.append(x[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))
                self.dx.append(dx[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))
                self.ddx.append(ddx[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))

        self.x = torch.stack(self.x)
        self.dx = torch.stack(self.dx)
        self.ddx = torch.stack(self.ddx)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.cat([self.x[idx], self.dx[idx]], dim=-1), torch.cat([self.dx[idx], self.ddx[idx]], dim=-1), torch.cat([self.dx[idx], self.ddx[idx]], dim=-1)


class LowDimPendulumDataset(Dataset):
    def __init__(self, path=f'{data_path}/pendulum', n_timesteps=2, mode='train'):
        super().__init__()
        try:
            print(f'Loading existing low-dim pendulum {mode} data...')
            x = torch.load(f'{path}/{mode}-z.pt')
            dx = torch.load(f'{path}/{mode}-dz.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating low-dim pendulum {mode} data...')
            n_ics = 200 if mode == 'train' else 20
            data = get_low_dim_pendulum_data(n_ics=n_ics)
            x = data['z'].reshape(n_ics, -1, data['z'].shape[-1])
            dx = data['dz'].reshape(n_ics, -1, data['dz'].shape[-1])
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/{mode}-z.pt')
            torch.save(dx, f'{path}/{mode}-dz.pt')

        self.n_timesteps = n_timesteps
        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = []
        self.dx = []
        for i in range(n_ics):
            for j in range(n_steps-n_timesteps):
                self.x.append(x[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))
                self.dx.append(dx[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))

        self.x = torch.stack(self.x)
        self.dx = torch.stack(self.dx)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dx[idx]


class LotkaVolterraDataset(Dataset):
    def __init__(self, path=f'{data_path}/lv', n_timesteps=2, interval=100, mode='train'):
        super().__init__()
        try:
            print(f'Loading existing Lotka-Volterra {mode} data...')
            x = torch.load(f'{path}/{mode}-z.pt')
            dx = torch.load(f'{path}/{mode}-dz.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating Lotka-Volterra {mode} data...')
            n_ics = 200 if mode == 'train' else 20
            x, dx = get_lv_data(n_ics=n_ics)
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/{mode}-z.pt')
            torch.save(dx, f'{path}/{mode}-dz.pt')

        self.n_timesteps = n_timesteps
        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = []
        self.dx = []
        for i in range(n_ics):
            for j in range(n_steps-n_timesteps*interval):
                self.x.append(x[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))
                self.dx.append(dx[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))

        self.x = torch.stack(self.x)
        self.dx = torch.stack(self.dx)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dx[idx]


class DoubleBump(torch.utils.data.Dataset):
    def __init__(self, path=f'{data_path}/doublebump', mode='train'):
        super().__init__()
        try:
            x = np.load(f'{path}/{mode}_x.npy')
            d = np.load(f'{path}/{mode}_d.npy')
        except FileNotFoundError:
            x, d = get_double_bump_data(n_samples=10000 if mode == 'train' else 1000)
            np.save(f'{path}/{mode}_x.npy', x)
            np.save(f'{path}/{mode}_d.npy', d)
        self.X, self.y = torch.FloatTensor(x), torch.FloatTensor(d)
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx], self.y[idx]  # no dx
