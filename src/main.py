# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 16:05:22 2025

@author: Eduardo
"""
import torch
from utilities import set_seed, treinamento, pca_latent_space, tSNE_2D, tSNE_3D, interpola_latente, cluster_latente, DBSCAN_latente
from dataset import Diagset_20x
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VAE_I2DL


def main():
    Z_DIM = 256
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-4
    set_seed(2025)
    AMOSTRA_FOLDER = 'xxxxx'
    VAL_FOLDER = 'xxxx'
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    
    train_dataset = Diagset_20x(AMOSTRA_FOLDER,
                                transform=transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    validation_dataset = Diagset_20x(VAL_FOLDER,
                                     transform=transform)
    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=BATCH_SIZE,
                                       shuffle=False)
    
    model = VAE_I2DL(Z_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=LEARNING_RATE)
    
    treinamento(model,
                NUM_EPOCHS,
                train_dataloader,
                optimizer,
                DEVICE)
    
    
    #AVALIAÇÃO
    model.eval()
    pca_latent_space(model, validation_dataloader, DEVICE)
    tSNE_2D(model, validation_dataloader, DEVICE)
    tSNE_3D(model, validation_dataloader, DEVICE)
    interpola_latente(model, validation_dataloader, DEVICE)
    cluster_latente(6, model, validation_dataloader, DEVICE)
    DBSCAN_latente(model, validation_dataloader, DEVICE)
    
if __name__ == '__main__':
    main()