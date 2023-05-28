
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
                                depth_size=args.pos3d_depth, 
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
                                depth_size=args.pos3d_depth, 
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
    if args.skip_model:
        model = SkipDPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                                num_seq_knowledge_source=args.num_seq_knowledge_source,
                                                pos3d_encoding=args.pos3d_encoding,
                                                pos3d_depth=args.pos3d_depth,
                                                initialize_ks_with_pos_embed=args.initialize_ks_with_pos_embed
                                                ).to(device)
    else:
        model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                                num_seq_knowledge_source=args.num_seq_knowledge_source,
                                                pos3d_encoding=args.pos3d_encoding,
                                                pos3d_depth=args.pos3d_depth,
                                                initialize_ks_with_pos_embed=args.initialize_ks_with_pos_embed,
                                                skip_step=args.skip_step
                                                ).to(device)

    lr_params = [
            {'params': model.neck.parameters()},
            {'params': model.head.parameters()},
            {'params': model.dpt.embeddings.parameters()},
            {'params': model.dpt.encoder.layer.parameters()},   
            {'params': model.dpt.layernorm.parameters()},      

            {'params': model.dpt.pos3d_encoder.parameters(), 'lr':args.lr_ext},
            {'params':  model.knowledge_sources.parameters(), 'lr':args.lr_ext},     
        ]
    if args.skip_model: lr_params += [{'params': model.dpt.encoder.mid_ks_layer.parameters(), 'lr':args.lr_ext}]
    optimizer = torch.optim.Adam(lr_params, lr=args.lr, weight_decay=2e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    # load if any checkpoint exists
    start_epoch = 0
    if not args.restart:
        start_epoch = checkpoint.load_checkpoint(args.output_dir, model, optimizer)

    criterion = create_loss(args.loss_fn, device)

    validate(args, model, dataloader_validation, criterion, device=device, step=0)

    model.train()
    if args.freeze_base:
        model.freeze_base()
    for epoch in range(start_epoch, args.epochs):
        
        step = epoch * len(dataloader_train)
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
                if args.freeze_base:
                    model.freeze_base()

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
    if args.lr_ext is None:
        args.lr_ext = args.lr
    main(args)
