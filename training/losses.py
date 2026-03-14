import torch


def gradient_penalty(discriminator, real, fake):
    batch_size = real.size(0)
    alpha_shape = [batch_size] + [1] * (real.dim() - 1)
    alpha = torch.rand(alpha_shape, device=real.device)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp