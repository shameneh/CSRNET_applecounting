import numpy as np
import torch


def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.225,0.225,0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def  weighted_mse_loss(inputs, targets, weights):  #return torch.sum(weights * (inputs - targets) ** 2)       
    loss = (inputs-targets)**2
    if weights is not None:
        loss *=weights.unsqueeze(2).unsqueeze(2).expand_as(loss)

    loss=torch.mean(loss)
    return loss
     

