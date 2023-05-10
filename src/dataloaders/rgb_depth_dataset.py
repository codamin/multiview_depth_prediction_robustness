import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def _gaussian_noise(x, scale=0.5):
    return x + torch.randn_like(x) * scale

def _gaussian_blur(x, sigma=5):
    k = 4 * sigma + 1
    return transforms.functional.gaussian_blur(x, kernel_size=k, sigma=sigma)

def _fog_3d(x, depth, fog_strength=150):
    depth = depth/depth.max()
    t = torch.exp(-fog_strength * depth)
    a = x.mean()
    return x * t + a * (1-t)

def _pixelate(x, resize=12):
    _, h, w = x.shape
    x = x[:,::resize,::resize]
    return F.interpolate(x[None,:], size=(h,w))[0]

def _identity(x):
    return x

transforms = {
    'gaussian_noise': _gaussian_noise,
    'gaussian_blur': _gaussian_blur,
    'fog_3d': _fog_3d,
    'pixelate': _pixelate,
    'identity': _identity,
}

class RGBDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_frames=16, image_size=384):
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms
        else:
            if isinstance(transform, str):
                transform = [transform]
            self.transform = {k: transforms[k] for k in transform}
        self.n_frames = n_frames
        self.inp_dir = os.path.join(root_dir, 'rgb')
        self.out_dir = os.path.join(root_dir, 'depth_zbuffer')
        self.inp_files = sorted(os.listdir(self.inp_dir))
        self.out_files = sorted(os.listdir(self.out_dir)) 

        self.resize_and_to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.n_seqs, self.list_n_frames, self.dict_inp_filename, self.dict_out_filename = self._get_num_seqs_frames()


    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        max_sample_idx = self.list_n_frames[idx] - self.n_frames
        
        # sample a point randomly between 0 and max_sample_idx
        sample_idx = np.random.randint(0, max_sample_idx)
        sample_idx = 0
        # generate a sequence of indices starting from sample_idx using pytorch
        seq_idx = torch.arange(sample_idx, sample_idx+self.n_frames)
        # get the filenames of the sequence and load the input images
        inp_filenames = [self.dict_inp_filename[idx][i] for i in seq_idx]
        inp_imgs = [Image.open(os.path.join(self.inp_dir, filename)) for filename in inp_filenames]
        inp_imgs = [self.resize_and_to_tensor(img) * 2 - 1 for img in inp_imgs]
        # load the output images
        out_filenames = [self.dict_out_filename[idx][i] for i in seq_idx]
        # load the mono channel images
        out_imgs = [Image.open(os.path.join(self.out_dir, filename)) for filename in out_filenames]
        out_imgs = [self.resize_and_to_tensor(img) for img in out_imgs]
        
        # apply transform
        transform_key = np.random.choiced(list(self.transform.keys()))
        if transform_key == 'fog_3d':
            inp_imgs = [self.transform[transform_key](img, depth) for img, depth in zip(inp_imgs, out_imgs)]
        else:
            inp_imgs = [self.transform[transform_key](img) for img in inp_imgs]

        inp_imgs = torch.stack(inp_imgs)
        out_imgs = torch.stack(out_imgs)

        # generate mask from out_imgs such that when one cell of out_imgs is 1, the corresponding cell in inp_imgs is False
        masks = torch.ones_like(out_imgs, dtype=torch.bool)
        masks[out_imgs == 65535.0] = 0

        out_imgs = 10000 / (out_imgs + 1e-05)

        return inp_imgs, out_imgs, masks


    def _get_num_seqs_frames(self):
        n_seqs = 0
        list_n_frames = []
        dict_inp_filename = {}
        dict_out_filename = {}
        for i, filename in enumerate(self.inp_files):
            seq = int(filename.split('_')[1])

            # check if key in dict
            if seq not in dict_inp_filename:
                dict_inp_filename[seq] = []
                dict_out_filename[seq] = []
            # add filename to dict
            dict_inp_filename[seq].append(filename)
            dict_out_filename[seq].append(self.out_files[i])

            if seq > n_seqs:
                n_seqs = seq
                n_frames = int(self.inp_files[i-1].split('_')[3])+1
                list_n_frames.append(n_frames)
                assert n_frames >= self.n_frames, "n_frames < self.n_frames"

        list_n_frames.append(int(self.inp_files[-1].split('_')[3])+1)

        return n_seqs+1, list_n_frames, dict_inp_filename, dict_out_filename
    
if __name__=="__main__":
    import matplotlib.pyplot as plt

    ds = RGBDepthDataset(root_dir='../../data', n_frames=2)
    img, depth, mask = ds[0]
    plt.imshow(img[0].permute(1,2,0))
    plt.show()