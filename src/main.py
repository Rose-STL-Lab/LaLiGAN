import torch
import numpy as np
from torch.utils.data import DataLoader
from gan import Discriminator, LieGenerator
from autoencoder import AutoEncoder
from sindy import SINDyRegression
from train import train_lassi
from utils import get_dataset
from parser_utils import get_args


if __name__ == '__main__':
    args = get_args()

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
    autoencoder = AutoEncoder(**args).to(args['device'])
    discriminator = Discriminator(**args).to(args['device'])
    generator = LieGenerator(**args).to(args['device'])
    args['regressor'] = SINDyRegression(**args).to(args['device']) if args['include_sindy'] else None

    # Train model
    train_lassi(autoencoder, discriminator, generator, train_loader, val_loader, **args)
    # Save final model
    torch.save(autoencoder.state_dict(), f'saved_models/{args["save_dir"]}/autoencoder.pt')
    torch.save(generator.getLi(), f'saved_models/{args["save_dir"]}/Lie_list.pt')
