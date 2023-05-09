
import wandb
import argparse
import yaml
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from src.models import DPTDepth
import src.checkpoint as checkpoint
from  src.losses import virtual_normal_loss, midas_loss

def get_args():
    config_parser = argparse.ArgumentParser()
    
    config_parser.add_argument('--config', default='', type=str)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_eval', default=16, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--save_weight_freq', default=5, type=int)
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
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

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
        vnl_loss = virtual_normal_loss.VNL_Loss(1.0, 1.0, (384, 384)).to(device)
        midas_loss = midas_loss.MidasLoss(alpha=0.1).to(device)
        def criterion(prediction, ground_truth, mask_valid):
            # Midas Loss
            _, ssi_loss, reg_loss = midas_loss(prediction, ground_truth, mask_valid)

            # Virtual Normal Loss
            vn_loss = vnl_loss(prediction, ground_truth)

            return ssi_loss + 0.1 * reg_loss + 10 * vn_loss
        return criterion
    else:
        raise Exception("Loss not implemented")


def main(args):

    dataloader_train = None
    dataloader_validation = None

    device = torch.device(args.device)

    # start a new wandb run to track this script
    wandb.init(
         # set the wandb project where this run will be logged
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_group,
    
        # track hyperparameters and run metadata
        config=args
    )

    # define the model
    model = DPTDepth.from_pretrained("Intel/dpt-large")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load if any checkpoint exists
    start_epoch = checkpoint.load_checkpoint(args.output_dir, model, optimizer)

    criterion = create_loss(args.loss_fn, device)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        step = epoch * len(dataloader_train)
        for x, depth, mask_valid in dataloader_train:

            optimizer.zero_grad()

            outputs = model(pixel_values=x)
            predicted_depth = outputs["predicted_depth"]

            loss_train = criterion(predicted_depth, depth, mask_valid)

            # backpropagate loss
            loss_train.backward()
            optimizer.step()

            # log metrics
            wandb.log({f"train loss ({args.loss_fn})": loss_train}, step=step)

            #eval every 10 steps
            if step % args.eval_freq == 0:
                validate(args, model, dataloader_validation, criterion, wandb, step)
                model.train()

            step += 1

@torch.no_grad()
def validate(args, model, dataloader_validation, criterion, wandb, step):

    losses = []
    original_images = []
    depth_images = []
    predicted_depths = []

    model.eval()
    for x, depth, mask_valid in dataloader_validation:

        outputs = model(pixel_values=x)
        predicted_depth = outputs["predicted_depth"]

        loss_val = criterion(predicted_depth, depth, mask_valid)
        losses.append(loss_val.item())

    # log metrics
    wandb.log({f"val loss ({args.loss_fn})": sum(losses)/len(losses)}, step=step)
    

if __name__=="__main__":
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
