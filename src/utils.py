import math
import os

import torch
import numpy as np
from PIL import Image

def _listify(x):
    x_list = []
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 4:
            x_list += [x[j] for j in range(x.shape[0])]
        elif len(x.shape) == 3:
            x_list += [x]
        else:
            raise Exception("Image shape unmatched")
    elif isinstance(x, list):
        for xi in x:
            if len(xi.shape) == 4:
                x_list += [xi[j] for j in range(xi.shape[0])]
            elif len(xi.shape) == 3:
                x_list += [xi]
            else:
                raise Exception("Image shape unmatched")
    return x_list

def _make_grid(images, rows, cols, mode):
    w, h = images[0].size
    grid = Image.new(mode, size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def rgb_tensor2PIL(x, return_list=False):
    x_list = _listify(x)
    images = []

    for xi in x_list:
        xi = xi.permute(1,2,0).clamp(-1,1).cpu()
        xi = (xi + 1) / 2
        images.append(Image.fromarray((255 * xi.numpy()).astype(np.uint8)))

    if return_list:
        return images
    
    n_images = len(images)
    return _make_grid(images, rows=math.ceil(n_images/5), cols=5, mode='RGB')

def depth_tensor2PIL(x, return_list=False):
    x_list = _listify(x)
    images = []

    for xi in x_list:
        xi = xi/xi.max()
        xi = xi[0].clamp(0,1).cpu()
        images.append(Image.fromarray((255 * xi.numpy()).astype(np.uint8)))

    if return_list:
        return images
    
    n_images = len(images)
    return _make_grid(images, rows=math.ceil(n_images/5), cols=5, mode='L')

def save_images(images, path, name):
    os.makedirs(os.path.join(path, 'samples'), exist_ok=True)
    if isinstance(images, list):
        for i, image in enumerate(images):
            image.save(os.path.join(path, 'samples', f'{name}:{i}.png'))
    else:
        images.save(os.path.join(path, 'samples', f'{name}.png'))

