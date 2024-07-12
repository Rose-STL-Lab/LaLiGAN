import torch
import numpy as np
from torch.utils.data import DataLoader
from autoencoder import AutoEncoder
from sindy import SINDyRegression
from train import train_SINDy
from parser_utils import get_sindy_args
from utils import get_dataset

if __name__ == '__main__':
    args = get_sindy_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # args to dict
    args = vars(args)

    # Load dataset
    train_dataset, val_dataset, args = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Initialize model
    if args['load_ae']:
        autoencoder = AutoEncoder(**args).to(args['device'])
        autoencoder.load_state_dict(torch.load(f'saved_models/{args["load_dir"]}/autoencoder.pt'))
    else:
        args['ae_arch'] = 'none'
        autoencoder = AutoEncoder(**args).to(args['device'])
    if args['load_Lie']:
        L_list = torch.load(f'saved_models/{args["load_dir"]}/Lie_list.pt')
        L_list = L_list[0].detach().cpu()
        args['L_list'] = [L_list[i] for i in range(L_list.shape[0])]
    else:
        args['L_list'] = []
    regressor = SINDyRegression(**args).to(args['device'])

    # Train regressor
    train_SINDy(autoencoder, regressor, train_loader, val_loader, **args)
