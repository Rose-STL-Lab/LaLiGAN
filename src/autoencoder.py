import torch.nn as nn
from torch.autograd.functional import jvp
from torch.nn.utils.parametrizations import orthogonal


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


class AutoEncoder(nn.Module):
    '''
    Arguments:
        input_dim: dimension of input
        hidden_dim: dimension of hidden layer
        latent_dim: dimension of latent layer
        n_layers: number of hidden layers
        n_comps: number of components
        activation: activation function
        flatten: whether to flatten input
    Input:
        x: (batch_size, n_comps, input_dim)
    Output:
        z: (batch_size, n_comps, latent_dim)
        xhat: (batch_size, n_comps, input_dim)
    '''
    def __init__(self, ae_arch, input_dim, hidden_dim, latent_dim, n_layers, n_comps=1, activation='ReLU', batch_norm=True, flatten=False, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten() if flatten else nn.Identity()
        if ae_arch == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                # Batch norm
                Reshape(-1, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                nn.BatchNorm1d(hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                Reshape(-1, n_comps, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                getattr(nn, activation)(*kwargs['activation_args']),
                *[nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    # Batch norm
                    Reshape(-1, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                    nn.BatchNorm1d(hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                    Reshape(-1, n_comps, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                    getattr(nn, activation)(*kwargs['activation_args']),
                ) for _ in range(n_layers-1)],
                nn.Linear(hidden_dim, latent_dim) if not kwargs['ortho_ae'] else orthogonal(nn.Linear(hidden_dim, latent_dim)),
                Reshape(-1, latent_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                nn.BatchNorm1d(latent_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                Reshape(-1, n_comps, latent_dim) if batch_norm and n_comps > 1 else nn.Identity(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                getattr(nn, activation)(*kwargs['activation_args']),
                *[nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    getattr(nn, activation)(*kwargs['activation_args']),
                ) for _ in range(n_layers-1)],
                nn.Linear(hidden_dim, input_dim),
            )
        elif ae_arch == 'conv1d':
            self.encoder = nn.Sequential(
                Reshape(-1, 1, input_dim),
                nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(int(input_dim/8)*64, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim) if not kwargs['ortho_ae'] else orthogonal(nn.Linear(32, latent_dim)),
                nn.BatchNorm1d(latent_dim) if batch_norm else nn.Identity(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, int(input_dim/8)*64),
                nn.Unflatten(1, (64, int(input_dim/8))),
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                Reshape(-1, input_dim),
                nn.Sigmoid(),
            )
        elif ae_arch == 'cnn':
            c = kwargs['cnn_chin']
            h = kwargs['cnn_chhid']
            self.encoder = nn.Sequential(
                Reshape(-1, c, input_dim, input_dim) if n_comps > 1 else nn.Identity(),
                nn.Conv2d(c, h//4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(h//4, h//2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(h//2, h, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(int(input_dim/8)*int(input_dim/8)*h, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim) if not kwargs['ortho_ae'] else orthogonal(nn.Linear(32, latent_dim)),
                nn.BatchNorm1d(latent_dim) if batch_norm else nn.Identity(),
                Reshape(-1, n_comps, latent_dim) if n_comps > 1 else nn.Identity(),
            )
            self.decoder = nn.Sequential(
                Reshape(-1, latent_dim) if n_comps > 1 else nn.Identity(),
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, int(input_dim/8)*int(input_dim/8)*h),
                nn.Unflatten(1, (h, int(input_dim/8), int(input_dim/8))),
                nn.ConvTranspose2d(h, h//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h//2, h//4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(h//4, c, kernel_size=3, stride=2, padding=1, output_padding=1),
                Reshape(-1, n_comps, c, input_dim, input_dim) if n_comps > 1 else nn.Identity(),
                nn.Sigmoid()
            )
        elif ae_arch == 'none':
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        z = self.encoder(x)
        xhat = self.decoder(z)
        return z, xhat

    def decode(self, z):
        return self.decoder(z)

    def compute_dz(self, x, dx):
        dz = jvp(self.encoder, x, v=dx)[1]
        return dz

    def compute_dx(self, z, dz):
        dx = jvp(self.decoder, z, v=dz)[1]
        return dx
