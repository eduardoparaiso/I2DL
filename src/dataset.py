# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:53:31 2025

@author: Eduardo
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Diagset_20x(Dataset):
  def __init__(self, root_dir, transform=None):
    self.data = [] # Amarzena imagens
    self.label = [] # Armazena os rotulos
    self.transform = transform  # Transformações opcionais a serem aplicadas às imagens

    # Mapeamento das classes (nomes de pastas) para valores inteiros
    self.label_map = {
      'N': 0,
      'R1': 1,
      'R2': 2,
      'R3': 3,
      'R4': 4,
      'R5': 5
    }

    # Listas temporárias para armazenar os caminhos dos arquivos e os rótulos correspondentes
    samples = []
    samples_labels = []
    
    # Percorre as subpastas no diretório raiz
    for class_folder in os.listdir(root_dir):
      class_path = os.path.join(root_dir, class_folder)
      
      # Ignora arquivos, processa apenas diretórios
      if not os.path.isdir(class_path):
        continue
        
      # Percorre os arquivos .npy dentro de cada subpasta
      for npy_file in os.listdir(class_path):
        if npy_file.endswith('.npy'):
          file_path = os.path.join(class_path, npy_file)
          samples.append(file_path)           # Armazena o caminho completo do arquivo .npy
          samples_labels.append(class_folder) # Armazena a classe associada a esse arquivo

    # Carrega os arquivos .npy em memória
    for sample_path, class_label in zip(samples, samples_labels):
      array = np.load(sample_path)  # Carrega o array numpy (esperado com shape [N, H, W, C])
      for patch in array:
        self.data.append(patch)  # Adiciona cada patch (imagem) individual ao dataset
        self.label.append(self.label_map[class_label])  # Converte o nome da classe para um valor inteiro
        
  
  # Retorna o tamanho total do dataset (número de patches)
  def __len__(self):
    return len(self.data)

  # Retorna um par (imagem, rótulo) dado um índice  
  def __getitem__(self, idx):
    patch = self.data[idx]  # Obtém o patch correspondente
    label = torch.tensor([self.label[idx]])  # Converte o rótulo em tensor (com 1 dimensão)


    # Aplica transformações, se fornecidas
    if self.transform:
      patch = self.transform(patch)

    return patch, label