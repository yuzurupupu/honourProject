import torch


def gradient_penalty(D, real, fake):

    alpha = torch.rand(real.size(0),1,1,1,1).cuda()

    interpolated = alpha*real + (1-alpha)*fake

    interpolated.requires_grad_(True)

    prob = D(interpolated)

    grad = torch.autograd.grad(
        outputs=prob,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob),
        create_graph=True
    )[0]

    grad = grad.view(grad.size(0),-1)

    gp = ((grad.norm(2,dim=1)-1)**2).mean()

    return gp