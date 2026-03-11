import torch


def save_checkpoint(models,optimizers,epoch,path):

    state = {

        "epoch":epoch,

        "models":{
            name:m.state_dict()
            for name,m in models.items()
        },

        "optimizers":{
            name:o.state_dict()
            for name,o in optimizers.items()
        }

    }

    torch.save(state,path)