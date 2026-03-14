g_iter = 1
d_iter = 1
cd_iter = 1
TOTAL_ITER = 200

gen_load = inf_train_gen(train_loader)

for iteration in range(TOTAL_ITER):
    for p in D.parameters():
        p.requires_grad = False
    for p in CD.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = True
    for p in G.parameters():
        p.requires_grad = True

    # Train Encoder + Generator
    for iters in range(g_iter):
        G.zero_grad()
        E.zero_grad()

        real_images = next(gen_load)
        _batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=True)

        z_rand = torch.randn((_batch_size, latent_dim), device=device)
        z_hat = E(real_images).view(_batch_size, -1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)

        c_loss = -CD(z_hat).mean()
        d_real_loss = D(x_hat).mean()
        d_fake_loss = D(x_rand).mean()
        d_loss = -d_fake_loss - d_real_loss
        l1_loss = 10 * criterion_l1(x_hat, real_images)
        loss1 = l1_loss + c_loss + d_loss

        if iters < g_iter - 1:
            loss1.backward()
        else:
            loss1.backward(retain_graph=True)

        e_optimizer.step()
        g_optimizer.step()
        g_optimizer.step()

    # Train D
    for p in D.parameters():
        p.requires_grad = True
    for p in CD.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False

    for iters in range(d_iter):
        d_optimizer.zero_grad()

        real_images = next(gen_load)
        _batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=True)

        z_rand = torch.randn((_batch_size, latent_dim), device=device)
        z_hat = E(real_images).view(_batch_size, -1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)

        x_loss2 = -2 * D(real_images).mean() + D(x_hat).mean() + D(x_rand).mean()
        gradient_penalty_r = calc_gradient_penalty(D, real_images.detach(), x_rand.detach())
        gradient_penalty_h = calc_gradient_penalty(D, real_images.detach(), x_hat.detach())

        loss2 = x_loss2 + gradient_penalty_r + gradient_penalty_h
        loss2.backward(retain_graph=True)
        d_optimizer.step()

    # Train CD
    for p in D.parameters():
        p.requires_grad = False
    for p in CD.parameters():
        p.requires_grad = True
    for p in E.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False

    for iters in range(cd_iter):
        cd_optimizer.zero_grad()

        z_rand = torch.randn((_batch_size, latent_dim), device=device)
        gradient_penalty_cd = calc_gradient_penalty(CD, z_hat.detach(), z_rand.detach())
        loss3 = -CD(z_rand).mean() - c_loss + gradient_penalty_cd

        loss3.backward(retain_graph=True)
        cd_optimizer.step()

    if iteration % 10 == 0:
        print(
            f"[{iteration}/{TOTAL_ITER}] "
            f"D: {loss2.item():.3f} "
            f"En_Ge: {loss1.item():.3f} "
            f"Code: {loss3.item():.3f}"
        )

        feat = np.squeeze((0.5 * real_images[0] + 0.5).detach().cpu().numpy())
        feat = nib.Nifti1Image(feat, affine=np.eye(4))
        plotting.plot_img(feat, title="X_Real")
        plotting.show()

        feat = np.squeeze((0.5 * x_hat[0] + 0.5).detach().cpu().numpy())
        feat = nib.Nifti1Image(feat, affine=np.eye(4))
        plotting.plot_img(feat, title="X_Recon")
        plotting.show()

        feat = np.squeeze((0.5 * x_rand[0] + 0.5).detach().cpu().numpy())
        feat = nib.Nifti1Image(feat, affine=np.eye(4))
        plotting.plot_img(feat, title="X_Rand")
        plotting.show()