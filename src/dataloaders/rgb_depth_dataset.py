import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RGBDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.n_frames = n_frames
        self.inp_dir = os.path.join(root_dir, 'rgb')
        self.out_dir = os.path.join(root_dir, 'depth_euclidean')
        self.inp_files = sorted(os.listdir(self.inp_dir))
        self.out_files = sorted(os.listdir(self.out_dir)) 

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
        inp_imgs = [transforms.ToTensor()(img) for img in inp_imgs]
        # load the output images
        out_filenames = [self.dict_out_filename[idx][i] for i in seq_idx]
        # load the mono channel images
        out_imgs = [Image.open(os.path.join(self.out_dir, filename)) for filename in out_filenames]
        out_imgs = [transforms.ToTensor()(img) for img in out_imgs]
        
        # apply transform
        if self.transform:
            inp_imgs = [self.transform(img) for img in inp_imgs]

        inp_imgs = torch.stack(inp_imgs)
        out_imgs = torch.stack(out_imgs) / 65535.0
        print(out_imgs[0,0,0,0])

        # generate mask from out_imgs such that when one cell of out_imgs is 1, the corresponding cell in inp_imgs is False
        masks = torch.ones_like(out_imgs, dtype=torch.bool)
        masks[out_imgs == 1.0] = 0

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