import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import so


class IntParameter(nn.Module):
    def __init__(self, k=2, noise=0.1):
        super(IntParameter, self).__init__()
        self.noise = noise
        self.k = k

    def forward(self, data):
        noise = torch.randn_like(data) * self.noise
        return torch.round(torch.clamp(self.k * (data + noise), -self.k - 0.49, self.k + 0.49))


class LieGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(LieGenerator, self).__init__()
        self.repr = kwargs['repr']
        self.uniform_max = kwargs['uniform_max']
        self.coef_dist = kwargs['coef_dist']
        # self.g_init = kwargs['g_init']
        self.task = kwargs['task']
        self.sigma_init = kwargs['sigma_init']
        self.int_param = kwargs['int_param']
        self.int_param_noise = kwargs['int_param_noise']
        self.int_param_max = kwargs['int_param_max']
        self.threshold = kwargs['gan_st_thresh']
        self.keep_center = kwargs['keep_center']
        self.activated_channel = None  # default to all channel
        self.construct_group_representation(self.repr)
        self.masks = [mask.to(kwargs['device']) if mask is not None else None for mask in self.masks]
        self.int_param_approx = IntParameter(k=self.int_param_max, noise=self.int_param_noise)

    def construct_group_representation(self, repr_str):
        # analyze the repr string
        repr = []
        tuple_list = repr_str.split(';')
        for t in tuple_list:
            t = t.strip()
            if t.startswith('(') and t.endswith(')'):
                elements = t[1:-1].split(',')
                elements = [e.strip() for e in elements]
                repr.append(tuple(elements))

        # repr is a tuple of (N1, N2, N3) indicating N1 N3-dim vectors acted on by N2-dim Lie group, (N1, STR) specifying the group info, or (N1,) indicating N1 scalars
        self.Li = nn.ParameterList()
        self.sigma = nn.ParameterList()
        self.masks = []  # mask for sequential thresholding
        self.n_comps = []
        self.n_channels = []
        self.learnable = []
        self.n_dims = 0
        for i, r in enumerate(repr):
            if len(r) == 3:
                n_comps, n_channels, n_dims = r
                n_comps, n_channels, n_dims = int(n_comps), int(n_channels), int(n_dims)
                Li = nn.Parameter(torch.randn(n_channels, n_dims, n_dims))
                mask = torch.ones_like(Li)
                self.Li.append(Li)
                self.masks.append(mask)
                self.n_comps.append(n_comps)
                self.n_channels.append(n_channels)
                self.learnable.append(True)
                self.n_dims += n_dims * n_comps
                self.sigma.append(nn.Parameter(torch.eye(n_channels, n_channels) * self.sigma_init, requires_grad=False))
            elif len(r) == 1:
                n_comps = int(r[0])
                self.Li.append(nn.Parameter(torch.zeros(1, n_comps, n_comps), requires_grad=False))
                self.masks.append(None)
                self.n_comps.append(1)
                self.n_channels.append(1)
                self.learnable.append(False)
                self.n_dims += n_comps
                self.sigma.append(nn.Parameter(torch.eye(1, 1)))
            elif len(r) == 2:
                n_comps, group_str = r
                n_comps = int(n_comps)
                self.masks.append(None)
                if group_str == 'so2':
                    self.Li.append(nn.Parameter(torch.FloatTensor([[[0.0, 1.0], [-1.0, 0.0]]]), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(1)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(1, 1) * self.sigma_init, requires_grad=False))
                elif group_str == 'so2*r':
                    self.Li.append(nn.Parameter(torch.FloatTensor([[[0.0, 1.0], [-1.0, 0.0]], [[0.1, 0.0], [0.0, 0.1]]]), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(2)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(2, 2) * self.sigma_init, requires_grad=False))
                elif group_str == 'so3':
                    self.Li.append(nn.Parameter(so(3), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(3)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 3
                    self.sigma.append(nn.Parameter(torch.eye(3, 3) * self.sigma_init, requires_grad=False))
                elif group_str == 'so3+1':
                    L = torch.zeros(3, 4, 4)
                    L[:, :3, :3] = so(3)
                    self.Li.append(nn.Parameter(L, requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(3)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 4
                    self.sigma.append(nn.Parameter(torch.eye(3, 3) * self.sigma_init, requires_grad=False))
                elif group_str == 'so4':
                    self.Li.append(nn.Parameter(so(4), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(6)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 4
                    self.sigma.append(nn.Parameter(torch.eye(6, 6) * self.sigma_init, requires_grad=False))
                else:
                    raise ValueError(f'Group {group_str} not implemented yet.')
            else:
                raise ValueError(f'Invalid representation string at position {i}: {r}')
        print(f'Constructed Lie group representation with {self.n_dims} latent dimensions.')

    def set_activated_channel(self, ch):
        self.activated_channel = ch

    def activate_all_channels(self):
        self.activated_channel = None

    def channel_corr(self):
        s = 0.0
        for Li in self.Li:
            norm = torch.einsum('kdf,kdf->k', Li, Li)
            Li_N = Li / (torch.sqrt(norm).unsqueeze(-1).unsqueeze(-1) + 1e-6)
            s += torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', Li_N, Li_N), diagonal=1)))
        return s

    def forward(self, x):  # random transformation on x
        # x: (batch_size, *, n_dims)
        # normalize x to have zero mean
        if not self.keep_center:
            x_mean = torch.mean(x, dim=list(range(len(x.shape)-1)), keepdim=True)
            x = x - x_mean
        batch_size = x.shape[0]
        output_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(batch_size, -1)
        # z = self.sample_coefficient(batch_size, x.device)
        # g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, self.getLi()))
        g_z = self.sample_group_element(batch_size, x.device)
        x_t = torch.einsum('bij,bj->bi', g_z, x)
        x_t = x_t.reshape(output_shape)
        if not self.keep_center:
            x_t = x_t + x_mean
        return x_t

    def set_threshold(self, threshold):
        # relative to max in each channel
        for Li, mask in zip(self.Li, self.masks):
            if mask is None:
                continue
            max_chval = torch.amax(torch.abs(Li), dim=(1, 2), keepdim=True)
            # mask.data = torch.logical_and(torch.abs(Li) > threshold * max_chval, mask).float()
            mask.data = (torch.abs(Li) > threshold * max_chval).float()

    def sample_group_element(self, batch_size, device):
        start_dim = 0
        g = []
        for Li, mask, sigma, n_comps, n_channels, learnable in zip(self.Li, self.masks, self.sigma, self.n_comps, self.n_channels, self.learnable):
            if learnable and self.int_param:
                Li = self.int_param_approx(Li)
            if learnable and mask is not None:
                Li = Li * mask
            z = self.sample_coefficient(batch_size, n_channels, sigma, device)
            g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, Li))
            for _ in range(n_comps):
                end_dim = start_dim + g_z.shape[1]
                g_z_padded = F.pad(g_z, (start_dim, self.n_dims - end_dim, start_dim, self.n_dims - end_dim))
                g.append(g_z_padded)
                start_dim = end_dim
        g = torch.stack(g, dim=1)
        g = torch.sum(g, dim=1)
        return g

    def sample_coefficient(self, batch_size, n_channels, params, device):
        if self.coef_dist == 'normal':
            sigma = params
            z = torch.randn(batch_size, n_channels, device=device) @ sigma
        elif self.coef_dist == 'uniform':
            uniform_max = params
            z = torch.rand(batch_size, n_channels, device=device) * 2 * uniform_max - uniform_max
        elif self.coef_dist == 'uniform_int_grid':
            uniform_max = params
            z = torch.randint(-int(uniform_max), int(uniform_max), (batch_size, n_channels), device=device, dtype=torch.float32)
        ch = self.activated_channel
        if ch is not None:  # leaving only specified columns
            mask = torch.zeros_like(z, device=z.device)
            mask[:, ch] = 1
            z = z * mask
        return z

    def transform(self, g_z, x, tp):
        return torch.einsum('bjk,bk->bj', g_z, x)
        # if tp == 'vector':
        #     return torch.einsum('bjk,btk->btj', g_z, x)
        # elif tp == 'scalar':
        #     return x
        # elif tp == 'grid':
        #     grid = F.affine_grid(g_z[:, :-1], x.shape)
        #     return F.grid_sample(x, grid)

    def getLi(self):
        # convert ParameterList to list of tensors
        return [self.int_param_approx(Li) if self.int_param and learnable
                else Li * mask if learnable else Li
                for Li, mask, learnable in zip(self.Li, self.masks, self.learnable)]


class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_comps, hidden_dim, n_layers, activation='ReLU', **kwargs):
        super(Discriminator, self).__init__()
        self.input_dim = latent_dim * n_comps
        if kwargs['use_original_x']:
            self.input_dim += kwargs['input_dim'] * n_comps
        if kwargs['use_invariant_y']:
            self.input_dim += kwargs['y_dim']
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            getattr(nn, activation)(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                getattr(nn, activation)(),
            ) for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, y=None, x=None):
        # z: latent representation; y: invariant label; x: original input
        z = z.reshape(z.shape[0], -1)
        if y is not None:
            z = torch.cat([z, y], dim=-1)
        if x is not None:
            x = x.reshape(x.shape[0], -1)
            z = torch.cat([z, x], dim=-1)
        validity = self.model(z)
        return validity
