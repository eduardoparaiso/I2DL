# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:14:01 2025

@author: Eduardo
"""

import numpy as np
from skimage.color import rgb2hed, hed2rgb
from matplotlib import pyplot as plt
















def visualize_rgb(hematoxiline, eosin, normalized=True):

  # Corrigir normalização: de [-1, 1] -> [0, 1]
  if normalized:
    h = (hematoxiline.squeeze().numpy() + 1) / 2
    e = (eosin.squeeze().numpy() + 1) / 2
  else:
    h = hematoxiline.squeeze().numpy()
    e = eosin.squeeze().numpy()


  null_channel = np.zeros_like(h)

  # Constrói as imagens HED
  hed_h = np.stack((h, null_channel, null_channel), axis=-1)
  hed_e = np.stack((null_channel, e, null_channel), axis=-1)
  hed_he = np.stack((h, e, null_channel), axis=-1)

  # Converte para RGB
  rgb_h = hed2rgb(hed_h)
  rgb_e = hed2rgb(hed_e)
  rgb_he = hed2rgb(hed_he)

  # Exibe lado a lado
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  titles = ['Hematoxilina (H)', 'Eosina (E)', 'H + E']
  images = [rgb_h, rgb_e, rgb_he]

  for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

  plt.tight_layout()
  plt.show()