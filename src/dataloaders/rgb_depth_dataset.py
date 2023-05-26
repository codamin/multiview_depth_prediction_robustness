import torch
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

torch.manual_seed(0)
np.random.seed(0)


def _gaussian_noise(x, scale=0.8, severity_idx=None):
    if severity_idx is not None:
        scale = [0.2, 0.5, 0.8, 1.1, 1.4][severity_idx]
    return torch.clip(x + torch.randn_like(x) * scale, min=-1, max=1)

def _gaussian_blur(x, sigma=8, severity_idx=None):
    if severity_idx is not None:
        sigma = [2, 5, 8, 11, 14][severity_idx]
    k = 4 * sigma + 1
    return transforms.functional.gaussian_blur(x, kernel_size=k, sigma=sigma)

def _fog_3d(x, depth, fog_strength=150, severity_idx=None):
    depth = depth/depth.max()
    if severity_idx is not None:
        fog_strength = [100, 125, 150, 175, 200][severity_idx]
    t = torch.exp(-fog_strength * depth)
    a = x.mean()
    return x * t + a * (1-t)

def _pixelate(x, resize=8, severity_idx=None):
    _, h, w = x.shape
    if severity_idx is not None:
        resize = [4, 6, 8, 10, 12][severity_idx]
    x = x[:,::resize,::resize]
    return F.interpolate(x[None,:], size=(h,w))[0]

def _identity(x, severity_idx=None):
    return x

corruptions = {
    'gaussian_noise': _gaussian_noise,
    'gaussian_blur': _gaussian_blur,
    'fog_3d': _fog_3d,
    'pixelate': _pixelate,
    'identity': _identity,
}

def get_initial_frustum(image_size, depth_size):
    _ = None
    xy = np.arange(-1,1, 2/image_size)
    zp = -np.arange(0,1, 1/depth_size)
    grid_x, grid_y = np.meshgrid(xy, xy, indexing='ij')
    x = (zp[:,_,_] * grid_x[_,:]).reshape(1, -1)
    y = (zp[:,_,_] * grid_y[_,:]).reshape(1, -1)
    z = (np.tile(zp[:,_,_], (1,image_size,image_size))).reshape(1, -1)
    return np.concatenate([x, y, z]).T

def load_camera_info(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

class RGBDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_frames=16, image_size=384, depth_size=5, train_set=True, small=False):
        self.root_dir = root_dir
        if transform is None:
            self.transform = corruptions
        else:
            if isinstance(transform, str):
                transform = [transform]
            self.transform = {k: corruptions[k] for k in transform}
        self.n_frames = n_frames
        self.image_size = image_size
        self.train_set = train_set
        self.small = small
        self.inp_dir = os.path.join(root_dir, 'rgb')
        self.out_dir = os.path.join(root_dir, 'depth_zbuffer')
        self.mask_dir = os.path.join(root_dir, 'mask_valid')
        self.point_dir = os.path.join(root_dir, 'point_info')
        self.inp_files = sorted(os.listdir(self.inp_dir), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3])))
        self.out_files = sorted(os.listdir(self.out_dir), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3])))
        self.mask_files = sorted(os.listdir(self.mask_dir), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3]))) 
        self.point_files = sorted(os.listdir(self.point_dir), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3]))) if os.path.exists(self.point_dir) else None

        self.resize_and_to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.n_seqs, self.list_n_frames, self.dict_inp_filename, self.dict_out_filename, self.dict_mask_filename, self.dict_point_filename = self._get_num_seqs_frames()
        self.initial_frustum = get_initial_frustum(image_size, depth_size)


    def __len__(self):
        if self.small:
            return self.n_seqs // 10
        return self.n_seqs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        max_sample_idx = self.list_n_frames[idx] - self.n_frames + 1
        
        # sample a point randomly between 0 and max_sample_idx
        if self.train_set:
            sample_idx = np.random.randint(0, max_sample_idx)
        else:
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

        point_filenames = [self.dict_point_filename[idx][i] for i in seq_idx]
        point_info = [load_camera_info(os.path.join(self.point_dir, filename)) for filename in point_filenames if filename is not None]
        if len(point_info) != 0:
            first_view_dict = point_info[0]
            inp_points = [self._get_camera_frustum(view_dict, first_view_dict) for view_dict in point_info]
        else:
            inp_points = torch.nan

        # mask_filenames = [self.dict_mask_filename[idx][i] for i in seq_idx]
        # # load the mono channel images
        # masks = [Image.open(os.path.join(self.mask_dir, filename)) for filename in mask_filenames]
        # masks = [(self.resize_and_to_tensor(img) < 0.99) for img in masks]
        
        # apply transform
        transform_key = np.random.choice(list(self.transform.keys()))
        severity_idx_values = np.random.randint(low=0, high=5, size=len(inp_imgs))
        if transform_key == 'fog_3d':
            inp_imgs = [self.transform[transform_key](img, depth, severity_idx=siv) for img, depth, siv in zip(inp_imgs, out_imgs, severity_idx_values)]
        else:
            inp_imgs = [self.transform[transform_key](img, severity_idx=siv) for img, siv in zip(inp_imgs, severity_idx_values)]

        inp_imgs = torch.stack(inp_imgs)
        out_imgs = torch.stack(out_imgs)
        inp_points = torch.stack(inp_points)
        # masks = torch.stack(masks)

        masks = torch.ones_like(out_imgs, dtype=torch.bool)
        masks[out_imgs == 65535.0] = 0

        out_imgs = 10000 / (out_imgs + 1e-05)

        return inp_imgs, out_imgs, inp_points, masks


    def _get_num_seqs_frames(self):
        n_seqs = 0
        list_n_frames = []
        dict_inp_filename = {}
        dict_out_filename = {}
        dict_mask_filename = {}
        dict_point_filename = {}
        for i, filename in enumerate(self.inp_files):
            seq = int(filename.split('_')[1])

            # check if key in dict
            if seq not in dict_inp_filename:
                dict_inp_filename[seq] = []
                dict_out_filename[seq] = []
                dict_point_filename[seq] = []
                dict_mask_filename[seq] = []
            # add filename to dict
            dict_inp_filename[seq].append(filename)
            dict_out_filename[seq].append(self.out_files[i])
            dict_mask_filename[seq].append(self.mask_files[i])
            dict_point_filename[seq].append(self.point_files[i] if self.point_files is not None else None)

            if seq > n_seqs:
                n_seqs = seq
                n_frames = int(self.inp_files[i-1].split('_')[3])+1
                list_n_frames.append(n_frames)
                assert n_frames >= self.n_frames, "n_frames < self.n_frames"

        list_n_frames.append(int(self.inp_files[-1].split('_')[3])+1)

        return n_seqs+1, list_n_frames, dict_inp_filename, dict_out_filename, dict_mask_filename, dict_point_filename
    
    def _get_camera_frustum(self, view_dict, first_view_dict):
        location = np.array(view_dict['camera_location'])
        rotation = np.array(view_dict['camera_rotation_final'])
        
        first_view_location = np.array(first_view_dict['camera_location'])
        first_view_rotation = np.array(first_view_dict['camera_rotation_final'])

        output = R.from_euler('xyz', first_view_rotation).inv().apply(
                R.from_euler('xyz', rotation).apply(self.initial_frustum)
                ) + (location - first_view_location)
        output = output.T.reshape(-1, self.image_size, self.image_size)
        return torch.tensor(output).float() / 100
    
if __name__=="__main__":
    import matplotlib.pyplot as plt

    ds = RGBDepthDataset(root_dir='/scratch/izar/aasadi/dataset/data/train', n_frames=4, small=True)
    img, depth, points, mask = ds[10]
    print(len(ds))
    print(mask.shape, img.shape)
    # plt.imshow((img[0] / 2 + 0.5).permute(1,2,0))
    # plt.show()