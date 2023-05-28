import torch
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import pandas as pd

import wandb
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import DPTMultiviewDepth, SkipDPTMultiviewDepth
from src.dataloaders import RGBDepthDataset
from src.losses import virtual_normal_loss, midas_loss

import src.checkpoint as checkpoint
import src.utils as utils

def get_args():
    config_parser = argparse.ArgumentParser()
    
    config_parser.add_argument('--config', default='', type=str)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--test_data_path', default=None, type=str)
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_eval', default=16, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--small', default=False, action='store_true')
    parser.set_defaults(small=False)

    parser.add_argument('--num_seq_knowledge_source', default=200, type=int)
    parser.add_argument('--initialize_ks_with_pos_embed', default=False, action='store_true')
    parser.set_defaults(initialize_ks_with_pos_embed=False)
    parser.add_argument('--pos3d_encoding', default=True, action='store_true')
    parser.add_argument('--no_pos3d_encoding', action='store_false', dest='pos3d_encoding')
    parser.set_defaults(pos3d_encoding=True)
    parser.add_argument('--pos3d_depth', default=5, type=int)
    parser.add_argument('--skip_model', default=False, action='store_true')
    parser.add_argument('--no_skip_model', action='store_false', dest='skip_model')
    parser.set_defaults(skip_model=False)
    parser.add_argument('--skip_step', default=4, type=int)

    parser.add_argument('--corruptions', default=None, type=str)
    parser.add_argument('--eval_corruptions', default=None, type=str)
    parser.add_argument('--n_frames', default=10, type=int)
    parser.add_argument('--img_size', default=384, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--save_weight_freq', default=5, type=int)
    parser.add_argument('--restart', default=True, action='store_true')
    parser.add_argument('--no_restart', action='store_false', dest='restart')
    parser.set_defaults(restart=True)
    parser.add_argument('--output_dir', default='results/')
    parser.add_argument('--test_output_dir', default=None, type=str)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_ext', type=float, default=None)
    parser.add_argument('--loss_fn', default='midas', type=str)
    parser.add_argument('--freeze_base', default=False, action='store_true')
    parser.add_argument('--no_freeze_base', action='store_false', dest='freeze_base')
    parser.set_defaults(freeze_base=False)

    parser.add_argument('--log_wandb', default=False, action='store_true')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default="multiview-robustness-cs-503", type=str)
    parser.add_argument('--wandb_entity', default="aav", type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    else:
        print('No config file specified. Using default arguments.')
        
    args = parser.parse_args(remaining)
    
    return args

@torch.no_grad()
def test(args, model, dataloader_validation, criterion, step, device, n_images=20):

    losses = []
    original_images = []
    depth_images = []
    predicted_depths = []

    model.eval()

    with tqdm(total=len(dataloader_validation)) as progress_bar:
        for x, depth, _, mask_valid in tqdm(dataloader_validation):

                x = x.reshape(-1, *x.shape[-3:]).to(device)
                depth = depth.reshape(-1, *depth.shape[-3:]).to(device)
                mask_valid = mask_valid.reshape(-1, *mask_valid.shape[-3:]).to(device)

                outputs = model(pixel_values=x)
                predicted_depth = outputs["predicted_depth"][:, None]

                loss_val = criterion(predicted_depth, depth, mask_valid)
                losses.append(loss_val.item())

                if len(original_images) < n_images:
                    original_images.append(x)
                    depth_images.append(depth)
                    predicted_depths.append(predicted_depth)
        progress_bar.update(1) # update progress
    

    # log metrics
    if args.log_wandb: wandb.log({f"val loss ({args.loss_fn})": sum(losses)/len(losses)}, step=step)

    pil_original_images = utils.rgb_tensor2PIL(original_images)
    pil_depth_images = utils.depth_tensor2PIL(depth_images)
    pil_predicted_images = utils.depth_tensor2PIL(predicted_depths)

    if args.log_wandb: utils.log_images(pil_original_images, pil_depth_images, pil_predicted_images, wandb)

    utils.save_images(pil_original_images, path=args.output_dir, name=f'{step:07d}_orig', mode='RGB')
    utils.save_images(pil_depth_images, path=args.output_dir, name=f'{step:07d}_depth', mode='L')
    utils.save_images(pil_predicted_images, path=args.output_dir, name=f'{step:07d}_prediction', mode='L')
    


class TestDataset(Dataset):
    def __init__(self, root_dir, n_frames, image_size):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        self.depth_size = 5
        self.initial_frustum = self.get_initial_frustum(self.image_size, self.depth_size)

        scenes = ['electro', 'playground', 'office', 'terrace', 'terrains', 'facade',
                  'pipes', 'courtyard', 'relief_2', 'meadow', 'delivery_area', 'kicker', 'relief'
                  ]

        self.image_paths = [[] for _ in range(len(scenes))]
        self.cam_translations = [[] for _ in range(len(scenes))]
        self.cam_rotations = [[] for _ in range(len(scenes))]

        for sidx, sname in scenes:
            data_path = f'{args.test_data_path}/{sname}/camera_pos.json'
            data = pd.read_json(data_path)
            for view_idx, view in data.iterrows():
                image_path = f'{type}/images/dslr_images/DSC_{view["id"]:04}.JPG'
                self.image_paths[sidx].append(image_path)
                self.cam_translations[sidx].append(np.array(view['tx'], view['ty'], view['tz']))
                self.cam_rotations[sidx].append(view['rx'], view['ry'], view['rz'])
            
        # choose n_frame views for each scene
        indices = np.random.choice(len(scenes), size=self.n_frames, replace=False)
        self.image_paths = [self.image_paths[idx] for idx in indices]
        self.cam_translations = [self.cam_translations[idx] for idx in indices]
        self.cam_rotations = [self.cam_rotations[idx] for idx in indices]

    def __getitem__(self, idx):
        images = Image.open(self.image_paths[idx])
        images = self.transform(images)
        point3d = self.get_camera_frustum(self.cam_translations[idx], self.cam_rotations[idx],
                                        self.cam_translations[0], self.cam_rotations[0])
        
        images = torch.stack(images)
        point3d = torch.stack(point3d)
        masks = torch.ones_like(images, dtype=torch.bool)

        return images, point3d, masks
    
    def get_initial_frustum(image_size, depth_size):
        _ = None
        xy = np.arange(-1,1, 2/image_size)
        zp = -np.arange(0,1, 1/depth_size)
        grid_x, grid_y = np.meshgrid(xy, xy, indexing='ij')
        x = (zp[:,_,_] * grid_x[_,:]).reshape(1, -1)
        y = (zp[:,_,_] * grid_y[_,:]).reshape(1, -1)
        z = (np.tile(zp[:,_,_], (1,image_size,image_size))).reshape(1, -1)
        return np.concatenate([x, y, z]).T

    def _get_camera_frustum(self, location, rotation, first_view_location, first_view_rotation):
        output = R.from_euler('xyz', first_view_rotation).inv().apply(
                R.from_euler('xyz', rotation).apply(self.initial_frustum)
                ) + (location - first_view_location)
        output = output.T.reshape(-1, self.image_size, self.image_size)
        return torch.tensor(output).float() / 100

    def __len__(self):
        return len(self.data)


if __name__=="__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    # load model
    model = SkipDPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                            num_seq_knowledge_source=args.num_seq_knowledge_source,
                                            pos3d_encoding=args.pos3d_encoding,
                                            pos3d_depth=args.pos3d_depth,
                                            initialize_ks_with_pos_embed=args.initialize_ks_with_pos_embed
                                            ).to(args.device)

    # load checkpoint
    model = checkpoint.load_checkpoint(args.output_dir, model, args.device)
    print(f"Loaded checkpoint from {args.output_dir}")

    # dataloader
    dataloader_test = DataLoader(dataset=RGBDepthDataset(
                            root_dir=args.data_path, 
                            n_frames=args.n_frames, 
                            image_size=args.img_size,
                        ),
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
        
    # model
    test(model, dataloader_test)