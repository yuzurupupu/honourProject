import os
import torch


def _get_state_dict(model):
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def save_checkpoint(models, optimizers, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "models": {
            name: _get_state_dict(m)
            for name, m in models.items()
        },
        "optimizers": {
            name: o.state_dict()
            for name, o in optimizers.items()
        }
    }

    torch.save(state, path)