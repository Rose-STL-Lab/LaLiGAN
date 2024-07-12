import torch
import numpy as np
from dataset import *

cos_corr = lambda x, y: torch.trace(x.T @ y) / torch.norm(x) / torch.norm(y)


# scale the tensor to have dummy position equal to 1
def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor


# so(n) Lie algebra
def so(n):
    L = np.zeros((n*(n-1)//2, n, n))
    k = 0
    for i in range(n):
        for j in range(i):
            L[k, i, j] = 1
            L[k, j, i] = -1
            k += 1
    return torch.tensor(L, dtype=torch.float32)


def get_dataset(args):
    if args['task'] == 'rd':
        train_dataset = ReactionDiffusionDataset(mode='train')
        val_dataset = ReactionDiffusionDataset(mode='val')
        args['input_dim'] = train_dataset[0][0].shape[0]
        args['flatten'] = False
    elif args['task'] == 'mt_rd':
        train_dataset = MultiTimestepReactionDiffusionDataset(mode='train')
        val_dataset = MultiTimestepReactionDiffusionDataset(mode='val')
        args['input_dim'] = train_dataset[0][0].shape[1]
    elif args['task'] == 'mt_rd_ds':
        train_dataset = MultiTimestepReactionDiffusionDataset(mode='train', downsample=True)
        val_dataset = MultiTimestepReactionDiffusionDataset(mode='val', downsample=True)
        args['input_dim'] = train_dataset[0][0].shape[1]
    elif args['task'] == 'ld_pendulum':
        train_dataset = LowDimPendulumDataset(mode='train', n_timesteps=args['n_comps'])
        val_dataset = LowDimPendulumDataset(mode='val', n_timesteps=args['n_comps'])
        args['input_dim'] = train_dataset[0][0].shape[1]
    elif args['task'] == 'lv':
        train_dataset = LotkaVolterraDataset(mode='train', n_timesteps=args['n_comps'])
        val_dataset = LotkaVolterraDataset(mode='val', n_timesteps=args['n_comps'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
    elif args['task'] == 'rot_shelf':
        train_dataset = RotatingShelfDataset(mode='train')
        val_dataset = RotatingShelfDataset(mode='test')
        args['input_dim'] = train_dataset[0][0].shape[1]
        args['ae_arch'] = 'cnn'
        args['cnn_chin'] = 3
        args['cnn_chhid'] = 64
    elif args['task'] == 'double_bump':
        train_dataset = DoubleBump(mode='train')
        val_dataset = DoubleBump(mode='test')
        args['input_dim'] = train_dataset[0][0].shape[-1]
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, args
