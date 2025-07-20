# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:50:23 2025

@author: Eduardo
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights



class Resnet_Encoder(nn.Module):
    """
    Encoder baseado na ResNet-50 pré-treinada para gerar vetores latentes (mu, logvar).
    Este encoder remove a última camada totalmente conectada da ResNet-50 e adiciona duas
    camadas lineares para estimar a média (mu) e o logaritmo da variância (logvar) da
    distribuição latente de cada entrada.

    Parâmetros:
        z_dim (int): dimensionalidade do espaço latente (tamanho de cada vetor z).
    """

    def __init__(self, z_dim=256):
        super(Resnet_Encoder, self).__init__()

        # Carrega a ResNet-50 com pesos pré-treinados do ImageNet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove a última camada totalmente conectada (classificadora)
        modules = list(resnet.children())[:-1]
        self.resnet_base = nn.Sequential(*modules)

        # Permite o fine-tuning da base da ResNet (opcional)
        for param in self.resnet_base.parameters():
            param.requires_grad = True

        # Camadas lineares para gerar os parâmetros da distribuição latente
        self.fc_mu = nn.Linear(resnet.fc.in_features, z_dim)       # média do vetor latente
        self.fc_logvar = nn.Linear(resnet.fc.in_features, z_dim)   # log variância do vetor latente
        

    def forward(self, x):
        """
        Processa uma entrada x e retorna os parâmetros da distribuição latente.
    
        Parâmetros:
            x: Tensor de entrada com shape [batch_size, 3, H, W]
    
        Retorna:
            mu: Tensor com a média dos vetores latentes (shape: [batch_size, z_dim])
            logvar: Tensor com o log da variância dos vetores latentes (shape: [batch_size, z_dim])
        """
        x = self.resnet_base(x)           # Extrai características convolucionais
        x = torch.flatten(x, 1)           # Achata o tensor (mantendo dimensão batch)
        mu = self.fc_mu(x)                # Gera a média do vetor latente
        logvar = self.fc_logvar(x)        # Gera o log da variância do vetor latente
        return mu, logvar   
    


class Decoder(nn.Module):
    """
    Este decoder recebe um vetor latente `z`, transforma esse vetor em um mapa de características
    inicial com `fc`, e o reconstrói progressivamente usando camadas transpostas
    (ConvTranspose2d) até obter uma imagem RGB de tamanho 224x224.

    Parâmetros:
        z_dim (int): dimensionalidade do espaço latente (tamanho de cada vetor z).
    """

    def __init__(self, z_dim=256):
        super(Decoder, self).__init__()

        # Camada totalmente conectada: transforma z em volume 3D inicial
        self.fc = nn.Linear(z_dim, 512 * 7 * 7)

        # Camadas deconvolucionais para upsampling progressivo
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28 -> 56x56
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56 -> 112x112
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 112x112 -> 224x224
            nn.Sigmoid()  # Normaliza saída para [0,1] (como uma imagem RGB)
        )

    def forward(self, x):
        """
        Reconstrói uma imagem a partir de um vetor latente.

        Parâmetros:
            x: Tensor com shape [batch_size, z_dim]

        Retorna:
            Tensor de imagem reconstruída com shape [batch_size, 3, 224, 224]
        """
        x = self.fc(x)                     # Transforma vetor z em volume 3D inicial
        x = x.view(-1, 512, 7, 7)          # Redimensiona para volume 3D
        x = self.deconv_layers(x)         # Reconstrói imagem com camadas transpostas
        return x




class VAE_I2DL(nn.Module):
    """
    Este modelo segue a estrutura típica de um VAE:
    - Um encoder que mapeia a imagem de entrada para os parâmetros (mu, logvar) de uma
      distribuição Gaussiana no espaço latente.
    - Um truque de reparametrização para permitir o backpropagation através da amostragem.
    - Um decoder que reconstrói a imagem a partir do vetor latente amostrado.

    Parâmetros:
        z_dim (int): dimensionalidade do espaço latente (tamanho de cada vetor z).
    """

    def __init__(self, z_dim=256):
        super(VAE_I2DL, self).__init__()

        # Encoder que extrai mu e logvar a partir de uma imagem de entrada
        self.encoder = Resnet_Encoder(z_dim)

        # Decoder que reconstrói a imagem a partir do vetor latente z
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        """
        Executa o fluxo completo do VAE: codificação, amostragem e decodificação.

        Parâmetros:
            x: Tensor de imagem de entrada com shape [batch_size, 3, 224, 224]

        Retorna:
            recon_x: imagem reconstruída
            mu: média da distribuição latente
            logvar: log da variância da distribuição latente
        """
        mu, logvar = self.encoder(x)              # Codifica entrada para parâmetros da distribuição
        z = self.reparameterize_trick(mu, logvar) # Amostra vetor latente z
        recon_x = self.decoder(z)                 # Reconstrói a imagem
        return recon_x, mu, logvar

    def reparameterize_trick(self, mu, logvar):
        """
        Aplica o truque de reparametrização para permitir o backpropagation através da amostragem.

        Parâmetros:
            mu: média da distribuição latente (tensor)
            logvar: log da variância (tensor)

        Retorna:
            z: vetor latente amostrado com ruído (mu + eps * std)
        """
        std = torch.exp(0.5 * logvar)           # Desvio padrão = sqrt(var) = exp(0.5 * logvar)
        eps = torch.randn_like(std)            # Ruído amostrado da N(0,1)
        z = mu + eps * std                     # Amostragem com reparametrização
        return z