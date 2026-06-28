import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Denoiser
from dataset import BMFRDataset, readExr
from tqdm import tqdm
import matplotlib.pyplot as plt

def tensor_to_img(tensor):
    cpu_img = tensor.detach().cpu().numpy()
    cpu_img = np.permute_dims(cpu_img, (0, 2, 3, 1))
    return cpu_img

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

    denoiser = Denoiser(9).to(device)
    denoiser.load_state_dict(torch.load("denoiser_weights.pth"))
    denoiser.eval()
    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)
        pred = denoiser(input)
        pred_hdr = pred
        img = tensor_to_img(pred_hdr)[0]
        target_img = tensor_to_img(target)[0]
        plt.imshow(img)
        plt.show()
        plt.imshow(target_img)
        plt.show()
        plt.imshow(tensor_to_img(input[:,:3,:,:])[0])
        plt.show()