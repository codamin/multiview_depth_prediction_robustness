
import wandb
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

    parser.add_argument('--data_path', default=None, type=str,
                        help='Path to the train dataset')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='Path to the validation dataset')

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_eval', default=16, type=int)
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

    parser.add_argument('--corruptions', default=None, type=str,
                        help='Specifies the corruption applied to the train data. If set to None then all corruption are randomly selected and applied. \
                        Available options are [gaussian_noise, gaussian_blur, fog_3d, pixelate, identity (for no corruption)] (default: %(default)s)')
    parser.add_argument('--eval_corruptions', default=None, type=str,
                        help='Specifies the corruption applied to the validation data. If set to None then all corruption are randomly selected and applied. \
                        Available options are [gaussian_noise, gaussian_blur, fog_3d, pixelate, identity (for no corruption)] (default: %(default)s)')
    parser.add_argument('--n_frames', default=10, type=int,
                        help='Number of frames loaded per scence for multiview training (default: %(default)s)')
    parser.add_argument('--img_size', default=384, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int,
                        help='Frequency of evaluation in steps (default: %(default)s)')
    parser.add_argument('--save_weight_freq', default=1000, type=int,
                        help='Frequency of saving weights in steps (default: %(default)s)')
    parser.add_argument('--restart', default=True, action='store_true',
                        help='Whether to restart training from the begining. If set to False the training is started from the latest checkpoint (default: %(default)s)')
    parser.add_argument('--no_restart', action='store_false', dest='restart')
    parser.set_defaults(restart=True)
    parser.add_argument('--output_dir', default='results/', 
                        help='Directory to store results and weights for the experiment (default: %(default)s)')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--loss_fn', default='midas', type=str, 
                        help='Loss function to use. Options are [midas, mse, l1] (default: %(default)s)')

    parser.add_argument('--log_wandb', default=False, action='store_true')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default="multiview-robustness-cs-503", type=str)
    parser.add_argument('--wandb_entity', default="aav", type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)

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
    model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                            num_seq_knowledge_source=args.num_seq_knowledge_source,
                                            pos3d_encoding=args.pos3d_encoding,
                                            pos3d_depth=args.pos3d_depth,
                                            ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    dataloader_train = DataLoader(
                            dataset=RGBDepthDataset(
                                root_dir=args.data_path, 
                                transform=args.corruptions,
                                n_frames=args.n_frames, 
                                image_size=args.img_size, 
                                depth_size=args.pos3d_depth, 
                                train_set=True,
                            ),
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                        )
    dataloader_validation = DataLoader(
                            dataset=RGBDepthDataset(
                                root_dir=args.eval_data_path,
                                transform=args.eval_corruptions,
                                n_frames=args.n_frames, 
                                image_size=args.img_size, 
                                depth_size=args.pos3d_depth, 
                                train_set=False,
                            ),
                            batch_size=args.batch_size_eval,
                            num_workers=args.num_workers,
                        )

    # load if any checkpoint exists
    start_step = None
    start_epoch = 0
    if not args.restart:
        start_step = checkpoint.load_checkpoint(args.output_dir, model, optimizer)
        start_epoch = start_step // len(dataloader_train)

    criterion = create_loss(args.loss_fn, device)

    validate(args, model, dataloader_validation, criterion, device=device, step=0)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        
        step = epoch * len(dataloader_train) if start_step is None else start_step
        start_step = None
        
        if args.log_wandb: wandb.log({f"Epoch": epoch}, step=step)
        for x, depth, camera_frustum, mask_valid in tqdm(dataloader_train, desc=f"Epoch {epoch}"):

            depth = depth.to(device)
            mask_valid = mask_valid.to(device)
            x = x.to(device)
            camera_frustum = camera_frustum.to(device)
            
            optimizer.zero_grad()
            
            #initial pass through model
            ks = None
            predicted_outputs = []
            depths = []
            masks = []
            inputs = []


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
def validate(args, model, dataloader_validation, criterion, step, device, n_images=10):

    losses = []
    original_images = []
    depth_images = []
    predicted_depths = []

    model.eval()

    with tqdm(total=len(dataloader_validation)) as progress_bar:
        for x, depth, camera_frustum, mask_valid in tqdm(dataloader_validation):

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
