import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to a configuration file")
    # Task & hyperparameters
    parser.add_argument('--task', type=str, default='rd', help='task to run')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr_ae', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--lr_d', type=float, default=1e-3, help='learning rate for discriminator')
    parser.add_argument('--lr_g', type=float, default=1e-3, help='learning rate for generator')
    parser.add_argument('--lr_sindy', type=float, default=1e-3, help='learning rate for SINDy')
    parser.add_argument('--w_recon', type=float, default=1, help='weight for reconstruction loss')
    parser.add_argument('--w_gan', type=float, default=1, help='weight for GAN loss')
    parser.add_argument('--reg_type', type=str, default='none', help='regularization type')
    parser.add_argument('--w_reg', type=float, default=1e-2, help='weight for regularization')
    parser.add_argument('--w_chreg', type=float, default=0.0, help='weight for multi-channel regularization')
    parser.add_argument('--w_sindy_z', type=float, default=1e-3, help='weight for SINDy loss in z')
    parser.add_argument('--w_sindy_x', type=float, default=1e-1, help='weight for SINDy loss in x')
    parser.add_argument('--sindy_reg_type', type=str, default='l1', help='regularization type')
    parser.add_argument('--w_sindy_reg', type=float, default=1e-1, help='weight for regularization')
    # General model configuration
    parser.add_argument('--latent_dim', type=int, default=2, help='latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers in autoencoder / discriminator')
    parser.add_argument('--n_comps', type=int, default=1, help='number of components in autoencoder input')
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function')
    parser.add_argument('--activation_args', nargs='+', type=float, default=[], help='arguments for activation function')
    # Autoencoder configuration
    parser.add_argument('--ae_arch', type=str, default='mlp', help='autoencoder architecture')
    parser.add_argument('--ortho_ae', action='store_true', help='use orthogonal parameterization for the final layer of the autoencoder')
    parser.add_argument('--batch_norm', action='store_true', help='use batch normalization')
    # Generator configuration
    parser.add_argument('--repr', type=str, default="(1,so2)", help='specify group representation acting on latent space')
    parser.add_argument('--coef_dist', type=str, default='normal', help='distribution of Lie algebra coefficients')
    parser.add_argument('--g_init', type=str, default='random', help='initialization of generator')
    parser.add_argument('--sigma_init', type=float, default=1, help='initialization of generator sampling variance')
    parser.add_argument('--uniform_max', type=float, default=1, help='max value for uniform distribution')
    parser.add_argument('--int_param', action='store_true', help='use integer parameters for generator')
    parser.add_argument('--int_param_max', type=int, default=2, help='max value for integer parameters')
    parser.add_argument('--int_param_noise', type=float, default=0.1, help='noise in integer reparameterization')
    parser.add_argument('--gan_st_freq', type=int, default=5, help='LieGAN sequential threshold frequency')
    parser.add_argument('--gan_st_thresh', type=float, default=0.3, help='LieGAN sequential threshold (relative to max)')
    parser.add_argument('--keep_center', action='store_true', help='keep center of latent space')
    # Discriminator configuration
    parser.add_argument('--use_original_x', action='store_true', help='original x as additional discriminator input')
    parser.add_argument('--use_invariant_y', action='store_true', help='invariant label as additional discriminator input')
    parser.add_argument('--y_dim', type=int, default=1, help='dimension of invariant label')
    # SINDy configuration
    parser.add_argument('--include_sindy', action='store_true', help='include SINDy in training the autoencoder & GAN')
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order')
    parser.add_argument('--include_sine', action='store_true', help='include sine terms')
    parser.add_argument('--include_exp', action='store_true', help='include exponential terms')
    parser.add_argument('--seq_thres_freq', type=int, default=100, help='sequential threshold frequency')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for sparsity')
    # Run settings
    parser.add_argument('--gpu', type=int, default=0, help='gpu to use, -1 for cpu')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')
    parser.add_argument('--save_interval', type=int, default=100, help='save interval')
    parser.add_argument('--print_li', action='store_true', help='print Lie algebra generator')
    parser.add_argument('--wandb_name', type=str, default='test', help='wandb run name')
    parser.add_argument('--save_dir', type=str, default='test', help='save directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args, _ = parser.parse_known_args()
    if args.config:
        args = parser.parse_args(parse_config(args.config))
    else:
        args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    return args


def get_sindy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to a configuration file")
    # Task & hyperparameters
    parser.add_argument('--task', type=str, default='rd', help='task to run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr_ae', type=float, default=1e-3, help='learning rate for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for regressor')
    parser.add_argument('--reg_type', type=str, default='l1', help='regularization type')
    parser.add_argument('--w_reg', type=float, default=1e-1, help='weight for regularization')
    parser.add_argument('--rel_loss', action='store_true', help='use relative loss')
    # General model configuration
    parser.add_argument('--latent_dim', type=int, default=2, help='latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers in autoencoder')
    parser.add_argument('--n_comps', type=int, default=1, help='number of components in autoencoder input')
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function')
    parser.add_argument('--activation_args', nargs='+', type=float, default=[], help='arguments for activation function')
    # Autoencoder configuration
    parser.add_argument('--ae_arch', type=str, default='mlp', help='autoencoder architecture')
    parser.add_argument('--ortho_ae', action='store_true', help='use orthogonal parameterization for the final layer of the autoencoder')
    parser.add_argument('--batch_norm', action='store_true', help='use batch normalization')
    # Load autoencoder & Lie algebra basis
    parser.add_argument('--load_ae', action='store_true', help='load autoencoder')
    parser.add_argument('--load_Lie', action='store_true', help='Lie algebra basis')
    parser.add_argument('--load_dir', type=str, default='autoencoder.pt', help='path to autoencoder')
    # SINDy configuration
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order')
    parser.add_argument('--include_sine', action='store_true', help='include sine terms')
    parser.add_argument('--include_exp', action='store_true', help='include exponential terms')
    parser.add_argument('--seq_thres_freq', type=int, default=100, help='sequential threshold frequency')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for sparsity')
    # Run settings
    parser.add_argument('--gpu', type=int, default=0, help='gpu to use, -1 for cpu')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')
    parser.add_argument('--save_interval', type=int, default=100, help='save interval')
    parser.add_argument('--wandb_name', type=str, default='sindy-test', help='wandb run name')
    parser.add_argument('--save_dir', type=str, default='sindy-test', help='save directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args, _ = parser.parse_known_args()
    if args.config:
        args = parser.parse_args(parse_config(args.config))
    else:
        args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    return args


def parse_config(file):
    file_path = "../configs/" + file + ".cfg"
    with open(file_path, 'r') as f:
        # Split lines, filter out empty lines and strip white spaces
        return [item.strip() for item in f.read().split() if item.strip()]
