import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Denoiser
from dataset import BMFRDataset, readExr
from tqdm import tqdm

def gradient_loss(pred, target):
    pred_dx, pred_dy = torch.gradient(pred, dim=(2, 3))
    target_dx, target_dy = torch.gradient(target, dim=(2, 3))

    loss_dx = torch.nn.L1Loss()(pred_dx, target_dx)
    loss_dy = torch.nn.L1Loss()(pred_dy, target_dy)

    return loss_dx + loss_dy

def combined_loss(pred, target):
    return torch.nn.L1Loss()(pred, target) #+ gradient_loss(pred, target) * 0.15

if __name__ == "__main__":
    device = device = torch.device('cpu')
    if torch.cuda.is_available(): device = torch.device('cuda:0')
    print(device)

    bmfrDataset = BMFRDataset()
    dataloader = DataLoader(
        bmfrDataset,
        batch_size=4,
        shuffle=True,
        num_workers=6,
        pin_memory=True,       # accélère le transfert CPU -> GPU
        drop_last=True,        # ignore le dernier batch s'il est incomplet (utile si batch_size ne divise pas exactement len(dataset))
    )

    initial_lr = 0.001

    denoiser = Denoiser(9).train().to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=initial_lr)
    criterion = combined_loss

    best_denoiser_dict = denoiser.state_dict()
    best_loss = 10000000
    num_epochs = 500
    for epoch in range(num_epochs):
        losses = 0
        for input, target in tqdm(dataloader):

            input = input.to(device)
            target = torch.log1p(target.to(device))
            pred = torch.log1p(denoiser(input))
            loss = combined_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        #t = float(epoch) / num_epochs
        lr = initial_lr / (1 + np.sqrt(epoch))
        #lr = initial_lr * np.exp(-epoch)
        for param in optimizer.param_groups:
            param["lr"] = lr

        losses /= len(dataloader)
        if losses < best_loss:
            best_loss = losses
            best_denoiser_dict = denoiser.state_dict().copy()
            torch.save(denoiser.state_dict(), "denoiser_weights.pth")
        print(f"loss: {losses:.4f}, best loss: {best_loss:.4f}, lr: {lr:.4f}")

    denoiser.eval()
    torch.save(denoiser.state_dict(), "denoiser_weights.pth")