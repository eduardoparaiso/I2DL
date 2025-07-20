# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:14:01 2025

@author: Eduardo
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

def set_seed(seed=2025):     
    """
    Define uma semente fixa para os geradores de números aleatórios do NumPy e PyTorch,
    garantindo reprodutibilidade nos experimentos.

    Parâmetros:
        seed (int): Valor da semente a ser usada.
    """
    
    np.random.seed(seed) # Define a semente do gerador de números aleatórios do NumPy       
    torch.manual_seed(seed) # Define a semente do gerador de números aleatórios do PyTorch (CPU)   
    torch.cuda.manual_seed(seed) # Define a semente do gerador de números aleatórios do PyTorch (GPU, via CUDA)


def kl_annealing(epoch, step=5, max_beta=1.0):
    """
    Controla o fator de ponderação (beta) da divergência KL durante o treinamento,
    com aumento gradual (annealing) ao longo dos epochs.

    Parâmetros:
        epoch (int): Época atual do treinamento.
        step (int): Número de épocas por incremento de beta. 
        max_beta (float): Valor máximo que beta pode atingir. 

    Retorna:
        beta (float): Valor atual de beta, limitado por max_beta.
    """
    # Calcula beta aumentando em etapas a cada 'step' épocas,
    # até atingir o valor máximo definido por max_beta.
    beta = min((epoch // step) * (max_beta / (100 // step)), max_beta)
    return beta


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Calcula a função de perda para um VAE, combinando erro de reconstrução 
    com a divergência KL ponderada por beta.

    Parâmetros:
        recon_x (Tensor): Saída reconstruída pelo decodificador.
        x (Tensor): Entrada original.
        mu (Tensor): Vetor de médias da distribuição latente.
        logvar (Tensor): Logaritmo da variância da distribuição latente.
        beta (float): Fator de ponderação da divergência KL (para annealing).

    Retorna:
        loss (float): Soma do erro de reconstrução e da KL divergência ponderada.
        BCE (float): Erro de reconstrução (Binary Cross-Entropy) médio por amostra.
        KLD (float): Termo de divergência KL ponderado por beta.
    """
    # Calcula o erro de reconstrução médio por amostra (BCE)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
    
    # Calcula a divergência KL média por amostra entre a distribuição latente e a normal padrão
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
    
    # Retorna a perda total, o BCE e o KLD ponderado
    return BCE + (beta * KLD), BCE, (beta * KLD)


def treinamento(model, num_epochs, train_dataloader, optimizer, device='cuda', annealing_step=5):
    """
    Esta função realiza o treinamento de um modelo que utiliza reconstrução e divergência KL,
    ajustando progressivamente o peso da KL via annealing.
    

    Parâmetros:
        model (nn.Module): modelo a ser treinado.
        num_epochs (int): Número total de épocas de treinamento.
        train_dataloader (DataLoader): Dataloader contendo os dados de treino
        optimizer (torch.optim.Optimizer): otimizador para atualizar os parâmetros do modelo.
        device (str): dispositivo para treinamento
        annealing_step (int): ritmo de crescimento do peso KL (beta).
    """
    
    for epoch in range(num_epochs):
        beta = kl_annealing(num_epochs, annealing_step) # Calcula o valor de beta com base na época atual        
        
        model.train() # Coloca o modelo em modo de treinamento
        
        # Inicializa os acumuladores de perda
        running_loss = 0
        running_bce = 0
        running_kl = 0
        

        for x, _ in tqdm(train_dataloader):           
            x = x.to(device)  # Move os dados para o dispositivo apropriado                     
            optimizer.zero_grad()  # Zera os gradientes do otimizador
            recon, mu, logvar = model(x) # Executa a passada direta no modelo
            
            loss, bce, beta_kl = loss_function(recon, x, mu, logvar, beta)  # Calcula a perda total, BCE e KL ponderado
        
            loss.backward()  # Calcula os gradientes          
            optimizer.step()  # Atualiza os pesos do modelo
            
            # Acumula os valores de perda para análise
            running_loss += loss.item()
            running_bce += bce.item()
            running_kl += beta_kl.item()
                    
            print(f"Epoch {epoch+1},  Loss: {loss.item():.4f},  BCE: {bce.item():.4f},  KL: {beta_kl.item():.4f} ------- Mu: {mu.mean().item():.4f} - Logvar: {logvar.mean().item():.4f}")
       
        if epoch % 10 == 0: 
            torch.save(model.state_dict(), 'vae_resnet50.pth')
    
    torch.save(model.state_dict(), 'vae_resnet50.pth')
    
    
def show_image_grid(images, n_rows=4, n_cols=4):
    """
    Mostra um grid de imagens com o número de linhas e colunas definidos.
    
    Parâmetros:
      images   – batch de imagens (tensor) no formato [B, C, H, W]
      n_rows   – número de linhas do grid
      n_cols   – número de colunas do grid
    """
    num_images = n_rows * n_cols
    images = images[:num_images]  # Garante que o número de imagens não exceda o grid

    grid = make_grid(images, nrow=n_cols)  # nrow = colunas
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))  # Converte de [C, H, W] para [H, W, C]
    plt.title("Amostras do DataLoader")
    plt.axis('off')
    plt.show()


def pca_latent_space(model, validation_dataloader, device='cuda'):
    """
    Coleta as representações latentes (vetores z) para os dados de validação,
    obtidos pelo encoder e reparametrização do modelo, e plota as duas primeiras dimensões
    desse espaço latente com cores correspondentes às classes (labels).

    Parâmetros:
        model (nn.Module): VAE que contém um encoder e a função de reparametrização.
        validation_dataloader (DataLoader): DataLoader contendo os dados de validação com labels.
        device (str): dispositivo para processamento.
    """
    
    latents = []  # lista para armazenar as representações latentes (z)
    labels = []   # lista para armazenar as labels correspondentes
    
    with torch.no_grad():  # desliga o cálculo do gradiente 
        for x, y in validation_dataloader:
            x = x.to(device)  # move dados para dispositivo (GPU/CPU)
            y = y.to(device)  # move labels para dispositivo
            
            # Passa os dados pelo encoder para obter média e log variância
            mu, logvar = model.encoder(x)
            
            # Aplica o truque de reparametrização para amostrar z
            z = model.reparameterize_trick(mu, logvar)
            
            # Armazena z e labels na CPU para evitar problemas de memória
            latents.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatena todas as amostras e labels e converte para numpy para plotagem
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # Plota um scatter plot das duas primeiras dimensões do espaço latente
    plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title('Espaço latente da VAE')
    plt.show()


def interpola_latente(model, validation_dataloader, device='cuda', interpolation_step=10):
    """
    A função extrai duas amostras latentes consecutivas do conjunto de validação e gera
    uma sequência interpolada entre elas. Para cada ponto interpolado, gera a reconstrução
    correspondente usando o decoder e plota as imagens resultantes

    Parâmetros:
        model (nn.Module):  modelo VAE contendo encoder e decoder.
        validation_dataloader (DataLoader): DataLoader contendo os dados de validação com labels.
        device (str): dispositivo para processamento.
        interpolation_step (int): número de passos na interpolação entre os dois vetores latentes.
    """

    with torch.no_grad():  # modo avaliação, sem cálculo de gradiente
        for x, y in validation_dataloader:
            x = x.to(device)
            y = y.to(device)

            # Obtém média e log variância do encoder
            mu, logvar = model.encoder(x)

            # Amostra z via truque de reparametrização
            z = model.reparameterize_trick(mu, logvar)
            break  # utiliza só o primeiro batch para interpolação

    # Seleciona os dois primeiros vetores latentes para interpolar
    z1 = z[0]
    z2 = z[1]

    # Gera valores alpha para interpolar linearmente entre z1 e z2
    alphas = torch.linspace(0, 1, steps=interpolation_step)

    interpolations = []
    for alpha in alphas:
        # Interpolação linear: z_interp = (1 - alpha)*z1 + alpha*z2
        z_interp = z1 * (1 - alpha) + z2 * alpha
        
        # Reconstrução da imagem a partir do vetor latente interpolado
        recon = model.decoder(z_interp.unsqueeze(0))  # adiciona dimensão batch
        interpolations.append(recon)

    # Cria figura para mostrar todas as imagens interpoladas lado a lado
    fig, axes = plt.subplots(1, len(interpolations), figsize=(15, 3))

    for i, recon in enumerate(interpolations):
        img = recon.squeeze(0)  # remove dimensão batch: [C, H, W]
        img = img.permute(1, 2, 0)  # reorganiza para [H, W, C] para plotagem

        # Caso imagem seja grayscale (canal único), remove a dimensão do canal
        if img.shape[2] == 1:
            img = img.squeeze(2)

        img = img.detach().cpu().numpy()

        # Exibe a imagem com cmap adequado
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis('off')
        axes[i].set_title(f'{i+1}')

    plt.suptitle('Interpolação no espaço latente')
    plt.show()


def tSNE_2D(model, validation_dataloader, device='cuda'):
    """
    A função extrai vetores latentes (z) das amostras no conjunto de validação, reduz a dimensionalidade
    com t-SNE para visualização bidimensional e plota o resultado com cores indicando as classes reais.

    Parâmetros:
        model(nn.Module): modelo VAE contendo encoder e método de reparametrização.
        validation_dataloader (DataLoader): conjunto de validação.
        device (str): dispositivo para processamento.
    """

    latents = []  # lista para armazenar as representações latentes (z)
    labels = []   # lista para armazenar as labels correspondentes
    
    with torch.no_grad():  # desliga o cálculo do gradiente 
        for x, y in validation_dataloader:
            x = x.to(device)  # move dados para o dispositivo
            y = y.to(device)  # move labels para o dispositivo
            
            # Obtém mu e logvar do encoder
            mu, logvar = model.encoder(x)
            
            # Aplica reparametrização para obter z
            z = model.reparameterize_trick(mu, logvar)
            
            # Armazena z e labels na CPU
            latents.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatena os vetores latentes e labels e converte para numpy
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # Aplica t-SNE para redução de dimensionalidade
    z_2d = TSNE(n_components=2).fit_transform(latents)
    
    # Cria gráfico de dispersão com cores por classe
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    
    # Gera legenda personalizada para as classes
    classes = np.unique(labels)
    handles = []
    for c in classes:
        handles.append(
            plt.Line2D([], [], marker='o', color='w', markerfacecolor=plt.cm.tab10(c / 10),
                       label=f'Classe {c}', markersize=10)
        )
    
    plt.legend(handles=handles, title="Classes")
    plt.title("t-SNE do espaço latente com legenda de classes")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()


def tSNE_3D(model, validation_dataloader, device='cuda'):
    """
    Esta função coleta os vetores latentes z a partir de dados de validação usando o encoder do modelo,
    aplica t-SNE para reduzir a dimensionalidade para 3 componentes e plota os pontos em um gráfico 3D
    com coloração baseada nos rótulos das classes.

    Parâmetros:
        model(nn.Module): modelo VAE contendo encoder e método de reparametrização.
        validation_dataloader (DataLoader): conjunto de validação.
        device (str): dispositivo para processamento.
    """

    latents = []  # Lista para armazenar as representações latentes (z)
    labels = []   # Lista para armazenar os rótulos correspondentes
    
    with torch.no_grad():  # Desliga o cálculo de gradientes (modo avaliação)
        for x, y in validation_dataloader:
            x = x.to(device)  # Move os dados para o dispositivo
            y = y.to(device)  # Move os rótulos para o dispositivo
            
            # Passa os dados pelo encoder para obter mu e logvar
            mu, logvar = model.encoder(x)
            
            # Aplica o truque de reparametrização para obter z
            z = model.reparameterize_trick(mu, logvar)
            
            # Armazena os vetores z e os rótulos (em CPU para posterior uso)
            latents.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatena todos os vetores e rótulos e converte para NumPy
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # Aplica t-SNE para redução de dimensionalidade para 3D
    tsne = TSNE(n_components=3)
    z_3d = tsne.fit_transform(latents)
    
    # Cria o gráfico 3D com os componentes do t-SNE
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cria o scatter plot com coloração baseada nos rótulos
    scatter = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                         c=labels, cmap='tab10', alpha=0.7)
    
    # Adiciona legenda automática com os rótulos das classes
    legend = ax.legend(*scatter.legend_elements(), title="Labels", loc='upper right')
    ax.add_artist(legend)
    
    # Configurações do gráfico
    ax.set_title("Espaço latente (t-SNE 3D)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    
    # Exibe o gráfico
    plt.show()


def cluster_latente(n_clusters, model, validation_dataloader, device='cuda'):
    """
       Esta função extrai vetores latentes z usando o encoder e reparametrização de um modelo VAE,
    aplica KMeans para agrupar essas representações em `n_clusters` grupos, e visualiza os clusters
    usando projeção t-SNE em duas dimensões.

    Parâmetros:
        n_clusters (int): número de grupos a ser formado pelo KMeans.
        model(nn.Module): modelo VAE contendo encoder e método de reparametrização.
        validation_dataloader (DataLoader): conjunto de validação.
        device (str): dispositivo para processamento.
    """

    num_clusters = n_clusters
    latents = []  # Lista para armazenar os vetores latentes (z)
    labels = []   # Lista para armazenar os rótulos reais (não usados na clusterização)
    
    with torch.no_grad():  # Modo avaliação: sem cálculo de gradientes
        for x, y in validation_dataloader:
            x = x.to(device)  # Move dados para o dispositivo
            y = y.to(device)  # Move rótulos para o dispositivo
            
            # Obtém mu e logvar do encoder
            mu, logvar = model.encoder(x)
            
            # Amostra z usando o truque de reparametrização
            z = model.reparameterize_trick(mu, logvar)
            
            # Armazena z e os rótulos em CPU
            latents.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatena todas as representações latentes e rótulos
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # Aplica o algoritmo KMeans para agrupar os vetores latentes
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(latents)  # cluster de cada amostra
    
    # Reduz a dimensionalidade para 2D com t-SNE para visualização
    z_2d = TSNE(n_components=2, random_state=42).fit_transform(latents)
    
    # Plota os clusters em 2D com cores distintas
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    
    # Barra de cores indicando os clusters
    plt.colorbar(scatter, label='Cluster')
    plt.title('Agrupamento KMeans no espaço latente (visualização t-SNE)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.show()
    
        
def DBSCAN_latente(model, validation_dataloader, eps=150, min_sample=5, device='cuda'):
    
    """
    A função extrai vetores latentes (z) de um modelo VAE treinado, padroniza esses vetores com `StandardScaler`,
    aplica DBSCAN para identificar agrupamentos com base em densidade e plota os clusters resultantes
    após projeção bidimensional com t-SNE.

    Parâmetros:
       model(nn.Module): modelo VAE contendo encoder e método de reparametrização.
       validation_dataloader (DataLoader): conjunto de validação.
       device (str): dispositivo para processamento.
       eps (float): distância máxima entre dois pontos para que sejam considerados vizinhos no DBSCAN.
       min_sample (int): número mínimo de amostras em um raio de `eps` para formar um núcleo no DBSCAN.

    """

    latents = []  # Lista para armazenar os vetores latentes (z)
    labels = []   # Lista para armazenar os rótulos reais (opcional, não usados aqui)
    
    with torch.no_grad():  # Desativa o cálculo de gradientes
        for x, y in validation_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Obtém parâmetros latentes do encoder
            mu, logvar = model.encoder(x)
            
            # Amostra z usando o truque de reparametrização
            z = model.reparameterize_trick(mu, logvar)
            
            # Armazena vetores latentes e rótulos
            latents.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatena todos os vetores latentes e rótulos
    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    # Padroniza os vetores latentes antes de aplicar DBSCAN
    Z_scaled = StandardScaler().fit_transform(latents)
    
    # Aplica DBSCAN no espaço latente escalado
    dbscan = DBSCAN(eps=eps, min_samples=min_sample)
    dbscan_labels = dbscan.fit_predict(Z_scaled)
    
    # Reduz para 2D com t-SNE para visualização
    Z_2D = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(Z_scaled)
    
    # Visualiza os clusters encontrados
    plt.figure(figsize=(8, 6))
    plt.scatter(Z_2D[:, 0], Z_2D[:, 1], c=dbscan_labels, cmap='tab10', s=5)
    plt.title("DBSCAN no espaço latente da VAE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster")
    plt.show()