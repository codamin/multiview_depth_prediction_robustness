import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import DPTMultiviewDepth
from src.dataloaders import RGBDepthDataset
from src.losses import virtual_normal_loss, midas_loss

import src.checkpoint as checkpoint
import src.utils as utils

def get_args():
    config_parser = argparse.ArgumentParser(description='YAML Configuration', add_help=True)
    
    config_parser.add_argument('--config', default='', type=str)

    parser = argparse.ArgumentParser(description='Final Configuration', add_help=True)

    parser.add_argument('--test_data_path', default=None, type=str,
                        help='Path to the test dataset')

    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--num_seq_knowledge_source', default=200, type=int,
                        help='Number of sequences to use as knowledge source at every layer (default: %(default)s)')
    parser.add_argument('--pos3d_encoding', default=True, action='store_true',
                        help='Whether to use positional encoding for 3D coordinates (default: %(default)s)')
    parser.add_argument('--no_pos3d_encoding', action='store_false', dest='pos3d_encoding')
    parser.set_defaults(pos3d_encoding=True)
    parser.add_argument('--pos3d_depth', default=5, type=int, 
                        help='Depth value D for sampling from [0,1] for the 3D coordinates (default: %(default)s)')
    parser.add_argument('--skip_step', default=4, type=int,
                        help='Number of layers to skip for injecting the knowledge source (default: %(default)s)')

    parser.add_argument('--img_size', default=384, type=int)

    parser.add_argument('--output_dir', default='results/', 
                        help='Directory to store results and weights for the experiment (default: %(default)s)')
    parser.add_argument('--device', default='cuda')

    try:
        args_config, remaining = config_parser.parse_known_args()
    except:
        parser.print_help()
        exit()
    
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    else:
        print('No config file specified. Using default arguments.')
        
    args = parser.parse_args(remaining)

    for arg in vars(args):
        attr = getattr(args, arg)
        if isinstance(attr, str) and attr.lower() == 'none':
            setattr(args, arg, None)
    
    return args

def create_loss(loss_fn, device, img_size):
    if loss_fn == 'mse':
        return lambda prediction, ground_truth, mask_valid: F.mse_loss(prediction, ground_truth)
    elif loss_fn == 'l1':
        return lambda prediction, ground_truth, mask_valid: F.l1_loss(prediction, ground_truth)
    elif loss_fn == 'midas':
        _vnl_loss = virtual_normal_loss.VNL_Loss(1.0, 1.0, (img_size, img_size)).to(device)
        _midas_loss = midas_loss.MidasLoss(alpha=0.1).to(device)
        def criterion(prediction, ground_truth, mask_valid):
            # Midas Loss
            _, ssi_loss, reg_loss = _midas_loss(prediction, ground_truth, mask_valid)

            # Virtual Normal Loss
            vn_loss = _vnl_loss(prediction, ground_truth)

            return ssi_loss + 0.1 * reg_loss + 10 * vn_loss
        return criterion
    else:
        raise Exception("Loss not implemented")


def main(args):
    device = torch.device(args.device)

    # define the model
    model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                                num_seq_knowledge_source=args.num_seq_knowledge_source,
                                                pos3d_encoding=args.pos3d_encoding,
                                                pos3d_depth=args.pos3d_depth,
                                                skip_step=args.skip_step
                                                ).to(device)
    
    dataloader_test = DataLoader(
                            dataset=RGBDepthDataset(
                                root_dir=args.test_data_path, 
                                n_frames=args.n_frames, 
                                image_size=args.img_size,  
                                train_set=False,
                            ),
                            batch_size=1,
                            num_workers=args.num_workers,
                        )

    checkpoint.load_checkpoint(args.output_dir, model)

    criterion = create_loss(args.loss_fn, device, args.img_size)

    test(args, model, dataloader_test, criterion, device=device, step=0)


@torch.no_grad()
def test(args, model, dataloader_test, criterion, step, device, n_images=10):

    losses = []
    original_images = []
    depth_images = []
    predicted_depths = []

    model.eval()

    with tqdm(total=len(dataloader_test)) as progress_bar:
        for x, depth, camera_frustum, mask_valid in tqdm(dataloader_test):

                depth = depth.to(device)
                mask_valid = mask_valid.to(device)
                x = x.to(device)
                camera_frustum = camera_frustum.to(device)

                ks = None
                inputs = []
                predicted_outputs = []
                depths = []
                masks = []

                for i in range(x.shape[1]):
                    outputs = model(pixel_values=x[:, i], knowledge_sources=ks, points3d=camera_frustum[:, i])
                    ks = outputs["knowledge_sources"]
                    predicted_depth = outputs["predicted_depth"][:, None]
                    inputs.append(x[:, i])
                    predicted_outputs.append(predicted_depth)
                    masks.append(mask_valid[:, i])
                    depths.append(depth[:, i])

                x = torch.cat(inputs, axis=0)
                predicted_depth = torch.cat(predicted_outputs, axis=0)
                mask_valid = torch.cat(masks, axis=0)
                depth = torch.cat(depths, axis=0)

                loss_val = criterion(predicted_depth, depth, mask_valid)
                losses.append(loss_val.item())

                if len(original_images) < n_images:
                    original_images.append(x)
                    depth_images.append(depth)
                    predicted_depths.append(predicted_depth)
        progress_bar.update(1) # update progress
    

    print(f"val loss ({args.loss_fn}): {sum(losses)/len(losses)}")

    pil_original_images = utils.rgb_tensor2PIL(original_images)
    pil_depth_images = utils.depth_tensor2PIL(depth_images)
    pil_predicted_images = utils.depth_tensor2PIL(predicted_depths)

    utils.save_images(pil_original_images, path=args.output_dir, name=f'{step:07d}_orig', mode='RGB', folder='test')
    utils.save_images(pil_depth_images, path=args.output_dir, name=f'{step:07d}_depth', mode='L', folder='test')
    utils.save_images(pil_predicted_images, path=args.output_dir, name=f'{step:07d}_prediction', mode='L', folder='test')
    

if __name__=="__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)