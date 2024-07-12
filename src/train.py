import torch
import torch.nn as nn
import numpy as np
import os

save_model_path = '../saved_models'


def train_lassi(
    autoencoder, discriminator, generator, train_loader, test_loader,
    num_epochs, lr_ae, lr_d, lr_g, w_recon, w_gan, reg_type, w_reg, w_chreg, use_original_x, gan_st_freq, gan_st_thresh, ae_arch,
    include_sindy, regressor, lr_sindy, w_sindy_z, w_sindy_x, sindy_reg_type, w_sindy_reg, seq_thres_freq, threshold,
    device, log_interval, save_interval, save_dir, **kwargs
):
    no_ae_flag = (ae_arch == 'none')
    if no_ae_flag:
        optimizer_ae = None
    else:
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    if include_sindy:
        optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr_sindy)
        scheduler_sindy = torch.optim.lr_scheduler.MultiStepLR(optimizer_sindy, milestones=[1, 2, 3], gamma=10)
        sindy_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()
    recon_loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        # torch.autograd.set_detect_anomaly(True)
        running_losses = [[], [], [], [], [], [], [], [], [], []]
        autoencoder.train()
        discriminator.train()
        generator.train()
        if include_sindy:
            regressor.train()
        for i, (x, dx, y) in enumerate(train_loader):
            x = x.to(device)
            if include_sindy:
                dx = dx.to(device)
            y = y.to(device) if kwargs['use_invariant_y'] else None
            bs = x.shape[0]

            # Adversarial ground truths
            valid = torch.ones((bs, 1)).to(device)
            fake = torch.zeros((bs, 1)).to(device)

            # Reconstruction loss
            z, xhat = autoencoder(x)
            loss_ae = w_recon * recon_loss(xhat, x)
            running_losses[0].append(loss_ae.item() / w_recon)
            loss_ae_rel = loss_ae / recon_loss(x, torch.zeros_like(x))
            running_losses[5].append(loss_ae_rel.item() / (w_recon + 1e-6))
            loss = loss_ae

            # Generator loss
            zt = generator(z)  # transformed latent space representation
            xt = autoencoder.decode(zt) if use_original_x else None
            xr = xhat if use_original_x else None
            d_fake = discriminator(zt, y, xt)
            loss_g = w_gan * adversarial_loss(d_fake, valid)
            running_losses[1].append(loss_g.item() / (w_gan + 1e-6))
            loss = loss + loss_g
            if reg_type == 'Lnorm':
                Ls = generator.getLi()
                loss_g_reg = -sum([torch.minimum(torch.norm(L, p=2, dim=None), torch.FloatTensor([np.prod(L.shape[:-1])]).to(device)) for L in Ls])
                # loss_g_reg = torch.norm(generator.getLi(), p=1, dim=None)
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'cosine':
                loss_g_reg = torch.abs(nn.CosineSimilarity(dim=-1)(zt, z).mean())
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'none':
                pass
            else:
                raise NotImplementedError
            loss_g_chreg = generator.channel_corr()
            running_losses[9].append(loss_g_chreg.item())
            loss = loss + w_chreg * loss_g_chreg

            # Discriminator loss
            z_detached = z.detach()
            zt_detached = zt.detach()
            xr_detached = xr.detach() if use_original_x else None
            xt_detached = xt.detach() if use_original_x else None
            loss_d_real = adversarial_loss(discriminator(z_detached, y, xr_detached), valid)
            loss_d_fake = adversarial_loss(discriminator(zt_detached, y, xt_detached), fake)
            running_losses[3].append(loss_d_real.item())
            running_losses[4].append(loss_d_fake.item())
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss = loss + loss_d

            # SINDy loss
            if include_sindy:
                dz = autoencoder.compute_dz(x, dx)
                dz_pred = regressor(z)
                dx_pred = autoencoder.compute_dx(z, dz_pred)
                loss_sindy_z = w_sindy_z * sindy_loss(dz_pred, dz)
                loss_sindy_x = w_sindy_x * sindy_loss(dx_pred, dx)
                running_losses[6].append(loss_sindy_z.item() / w_sindy_z)
                running_losses[7].append(loss_sindy_x.item() / w_sindy_x)
                loss = loss + loss_sindy_z + loss_sindy_x
                if sindy_reg_type == 'l1':
                    loss_sindy_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                    running_losses[8].append(loss_sindy_reg.item())
                    loss = loss + w_sindy_reg * loss_sindy_reg
                else:
                    raise ValueError(f'Unknown regularization type: {reg_type}')

            # Backprop
            if not no_ae_flag:
                optimizer_ae.zero_grad()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if include_sindy:
                optimizer_sindy.zero_grad()
            loss.backward()
            if not no_ae_flag:
                optimizer_ae.step()
            optimizer_d.step()
            optimizer_g.step()
            if include_sindy:
                optimizer_sindy.step()

        if include_sindy:
            scheduler_sindy.step()

        # sequential thresholding
        if gan_st_freq > 0 and (epoch + 1) % gan_st_freq == 0:
            generator.set_threshold(gan_st_thresh)
        if include_sindy and seq_thres_freq > 0 and (epoch + 1) % seq_thres_freq == 0:
            regressor.set_threshold(threshold)
            w_reg *= 0.5

        if (epoch + 1) % log_interval == 0:
            print(f'Epoch {epoch}, loss_ae: {np.mean(running_losses[0]):.4f}, loss_g: {np.mean(running_losses[1]):.4f}, loss_g_reg: {np.mean(running_losses[2]):.4f}, loss_g_chreg: {np.mean(running_losses[9]):.4f}, loss_d_real: {np.mean(running_losses[3]):.4f}, loss_d_fake: {np.mean(running_losses[4]):.4f}, loss_ae_rel: {np.mean(running_losses[5]):.4f}, loss_sindy_z: {np.mean(running_losses[6]):.4f}, loss_sindy_x: {np.mean(running_losses[7]):.4f}, loss_sindy_reg: {np.mean(running_losses[8]):.4f}')
            autoencoder.eval()
            discriminator.eval()
            generator.eval()
            with torch.no_grad():
                running_losses = [[], [], [], [], [], [], []]
                for i, (x, dx, y) in enumerate(test_loader):
                    x = x.to(device)
                    dx = dx.to(device)
                    y = y.to(device) if kwargs['use_invariant_y'] else None
                    bs = x.shape[0]
                    valid = torch.ones((bs, 1)).to(device)
                    fake = torch.zeros((bs, 1)).to(device)
                    z, xhat = autoencoder(x)
                    zt = generator(z)
                    xt = autoencoder.decode(zt)
                    d_fake = discriminator(zt, y, xt if use_original_x else None)
                    d_real = discriminator(z, y, xhat if use_original_x else None)
                    loss_ae = recon_loss(xhat, x)
                    loss_ae_rel = loss_ae / recon_loss(x, torch.zeros_like(x))
                    loss_g = adversarial_loss(d_fake, valid)
                    loss_d_real = adversarial_loss(d_real, valid)
                    loss_d_fake = adversarial_loss(d_fake, fake)
                    running_losses[0].append(loss_ae.item())
                    running_losses[1].append(loss_g.item())
                    running_losses[2].append(loss_d_real.item())
                    running_losses[3].append(loss_d_fake.item())
                    running_losses[4].append(loss_ae_rel.item())
                    if include_sindy:
                        dz = autoencoder.compute_dz(x, dx)
                        dz_pred = regressor(z)
                        dx_pred = autoencoder.compute_dx(z, dz_pred)
                        loss_sindy_z = sindy_loss(dz_pred, dz)
                        loss_sindy_x = sindy_loss(dx_pred, dx)
                        running_losses[5].append(loss_sindy_z.item())
                        running_losses[6].append(loss_sindy_x.item())

                print(f'Epoch {epoch} test, loss_ae: {np.mean(running_losses[0]):.4f}, loss_g: {np.mean(running_losses[1]):.4f}, loss_d_real: {np.mean(running_losses[2]):.4f}, loss_d_fake: {np.mean(running_losses[3]):.4f}, loss_ae_rel: {np.mean(running_losses[4]):.4f}, loss_sindy_z: {np.mean(running_losses[5]):.4f}, loss_sindy_x: {np.mean(running_losses[6]):.4f}')
                if kwargs['print_li']:
                    print(generator.getLi())
                if include_sindy:
                    regressor.print()

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'{save_model_path}/{save_dir}'):
                os.makedirs(f'{save_model_path}/{save_dir}')
            torch.save(autoencoder.state_dict(), f'{save_model_path}/{save_dir}/autoencoder_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'{save_model_path}/{save_dir}/discriminator_{epoch}.pt')
            torch.save(generator.state_dict(), f'{save_model_path}/{save_dir}/generator_{epoch}.pt')
            if include_sindy:
                torch.save(regressor.state_dict(), f'{save_model_path}/{save_dir}/regressor_{epoch}.pt')


def train_SINDy(
        autoencoder, regressor, train_loader, test_loader,
        num_epochs, lr, reg_type, w_reg, seq_thres_freq, threshold, rel_loss,
        device, log_interval, save_interval, save_dir, **kwargs
):
    # Initialize optimizers
    optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr)

    # Loss functions
    sindy_loss = torch.nn.MSELoss()
    recon_loss = torch.nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        regressor.train()
        running_losses = [[], [], [], []]
        for i, (x, dx, _) in enumerate(train_loader):
            x = x.to(device)
            dx = dx.to(device)
            # Regularization loss
            if reg_type == 'l1':
                loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                running_losses[0].append(loss_reg.item())
            else:
                raise ValueError(f'Unknown regularization type: {reg_type}')
            loss = w_reg * loss_reg

            # Reconstruction loss
            z, xhat = autoencoder(x)
            loss_recon = recon_loss(xhat, x)
            running_losses[1].append(loss_recon.item())
            # dz loss & dx loss
            dz = autoencoder.compute_dz(x, dx)
            dz_pred = regressor(z)
            dx_pred = autoencoder.compute_dx(z, dz_pred)
            if rel_loss:
                # Denominator at least 0.1
                denom = torch.max(sindy_loss(dz, torch.zeros_like(dz, device=device)),
                                  torch.ones_like(loss, device=device) * 0.1)
                loss_sindy_z = sindy_loss(dz_pred, dz) / denom
            else:
                loss_sindy_z = sindy_loss(dz_pred, dz)
            loss_sindy_x = sindy_loss(dx_pred, dx)
            running_losses[2].append(loss_sindy_z.item())
            running_losses[3].append(loss_sindy_x.item())
            loss += loss_sindy_z

            # Optimization
            optimizer_sindy.zero_grad()
            loss.backward()
            optimizer_sindy.step()

        # Sequential thresholding
        if seq_thres_freq > 0 and (epoch + 1) % seq_thres_freq == 0:
            regressor.set_threshold(threshold)
            w_reg *= 0.5

        if (epoch + 1) % log_interval == 0:
            print(f'Epoch {epoch}, loss_reg: {np.mean(running_losses[0]):.4f}, '
                  f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                  f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                  f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
            regressor.eval()
            autoencoder.eval()
            with torch.no_grad():
                running_losses = [[], [], [], []]
                for i, (x, dx, _) in enumerate(test_loader):
                    x = x.to(device)
                    dx = dx.to(device)
                    z, xhat = autoencoder(x)
                    loss_recon = recon_loss(xhat, x)
                    dz = autoencoder.compute_dz(x, dx)
                    dz_pred = regressor(z)
                    dx_pred = autoencoder.compute_dx(z, dz_pred)
                    loss_sindy_z = sindy_loss(dz_pred, dz)
                    loss_sindy_x = sindy_loss(dx_pred, dx)
                    running_losses[1].append(loss_recon.item())
                    running_losses[2].append(loss_sindy_z.item())
                    running_losses[3].append(loss_sindy_x.item())

                    # Regularization
                    if reg_type == 'l1':
                        loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                        running_losses[0].append(loss_reg.item())
                    else:
                        raise ValueError(f'Unknown regularization type: {reg_type}')

                print(f'Epoch {epoch} test, loss_reg: {np.mean(running_losses[0]):.4f}, '
                      f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                      f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                      f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
                regressor.print()

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'{save_model_path}/{save_dir}'):
                os.makedirs(f'{save_model_path}/{save_dir}')
            torch.save(regressor.state_dict(), f'{save_model_path}/{save_dir}/regressor_{epoch}.pt')
