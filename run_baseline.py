
import wandb
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import DPTDepth
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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_eval', default=16, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--small', default=False, action='store_true')
    parser.set_defaults(small=False)

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
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_fn', default='midas', type=str)

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

    args = parser.parse_args(remaining)

    return args

def create_loss(loss_fn, device):
    if loss_fn == 'mse':
        return lambda prediction, ground_truth, mask_valid: F.mse_loss(prediction, ground_truth)
    elif loss_fn == 'l1':
        return lambda prediction, ground_truth, mask_valid: F.l1_loss(prediction, ground_truth)
    elif loss_fn == 'midas':
        _vnl_loss = virtual_normal_loss.VNL_Loss(1.0, 1.0, (args.img_size, args.img_size)).to(device)
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

    dataloader_train = DataLoader(
                            dataset=RGBDepthDataset(
                                root_dir=args.data_path, 
                                n_frames=args.n_frames, 
                                image_size=args.img_size, 
                                train_set=True,
                                small=args.small,
                            ),
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                        )
    dataloader_validation = DataLoader(
                            dataset=RGBDepthDataset(
                                root_dir=args.eval_data_path, 
                                n_frames=args.n_frames, 
                                image_size=args.img_size,  
                                train_set=False,
                                small=args.small,
                            ),
                            batch_size=args.batch_size_eval,
                            num_workers=args.num_workers,
                        )

    device = torch.device(args.device)


    if args.log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
        
            # track hyperparameters and run metadata
            config=args
        )

    # define the model
    model = DPTDepth.from_pretrained("Intel/dpt-large").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    # load if any checkpoint exists
    start_epoch = 0
    if not args.restart:
        start_epoch = checkpoint.load_checkpoint(args.output_dir, model, optimizer)

    criterion = create_loss(args.loss_fn, device)

    validate(args, model, dataloader_validation, criterion, device=device, step=0)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        
        step = epoch * len(dataloader_train)
        if args.log_wandb: wandb.log({f"Epoch": epoch}, step=step)
        for x, depth, _, mask_valid in tqdm(dataloader_train, desc=f"Epoch {epoch}"):
            
            x = x.reshape(-1, *x.shape[-3:]).to(device)
            depth = depth.reshape(-1, *depth.shape[-3:]).to(device)
            mask_valid = mask_valid.reshape(-1, *mask_valid.shape[-3:]).to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=x)
            predicted_depth = outputs["predicted_depth"][:, None]

            loss_train = criterion(predicted_depth, depth, mask_valid)

            # backpropagate loss
            loss_train.backward()
            optimizer.step()

            # log metrics
            if args.log_wandb: wandb.log({f"train loss ({args.loss_fn})": loss_train}, step=step)

            #eval every 10 steps
            if step != 0 and step % args.eval_freq == 0:
                validate(args, model, dataloader_validation, criterion, step, device)
                model.train()

            step += 1

            scheduler.step()
            if (step + 1) % args.save_weight_freq == 0:
                checkpoint.save_checkpoint(args.output_dir, step+1, model, optimizer)
            

@torch.no_grad()
def validate(args, model, dataloader_validation, criterion, step, device, n_images=20):

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
    

if __name__=="__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
