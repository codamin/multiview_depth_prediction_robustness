
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from tqdm import tqdm
import yaml
import argparse

import torch
from torch.utils.data import DataLoader

from src.models import DPTMultiviewDepth
from src.dataloaders import RGBDepthDataset
from src.dataloaders import gaussian_blur, gaussian_noise, fog_3d, pixelate
import src.checkpoint as checkpoint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_args():
    config_parser = argparse.ArgumentParser(description='YAML Configuration', add_help=True)
    
    config_parser.add_argument('--config', default='', type=str)

    parser = argparse.ArgumentParser(description='Final Configuration', add_help=True)

    parser.add_argument('--test_data_path', default=None, type=str,
                        help='Path to the test dataset')

    parser.add_argument('--batch_size', default=16, type=int)
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

@torch.no_grad()
def calculate_attention_ratio(model, test_dl, info, save_path):

    sev = info['severities']
    corruption_fn = info['fn']
    name = info['name']
    idx = model.config.num_seq_knowledge_source

    for layer in [4, 8, 12, 16, 20]:
        attn_ratio = []
        for s in sev:
            fn = (lambda x, y: corruption_fn(x, y, s=s)) if name == 'fog' else (lambda x: corruption_fn(x, s=s))
            ar = []
            for x, depth, camera_frustum, mask_valid in tqdm(test_dl):
                    
                    x = x.to(device)
                    depth = depth.to(device)
                    camera_frustum = camera_frustum.to(device)
                    mask_valid = mask_valid.to(device)
        
                    outputs = model(pixel_values=x[:, 0], points3d=camera_frustum[:, 0])
                    ks = outputs["knowledge_sources"]

                    if name == 'fog':
                        xn = fn(x[:, 1], 10000 / depth[:, 1])
                    else:
                        xn = fn(x[:, 1])
        
                    outputs = model(pixel_values=xn, knowledge_sources=ks, points3d=camera_frustum[:, 1], output_attentions=True)
                    attn = outputs["attentions"]
                    attn = attn[layer].mean(dim=[0,1])
                    ar.append((attn[:-idx, -idx:].sum()/attn[:-idx, :-idx].sum()).cpu())
                
            attn_ratio.append(sum(ar)/len(ar))
        plt.plot(sev, attn_ratio, label=f"layer: {layer}")

    plt.xlabel(info['label'])
    plt.ylabel("Attention Ratio")
    plt.legend(bbox_to_anchor=(1.01, 1.0))
    plt.savefig(f'{save_path}/attn_ratio_{name}.pdf', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    args = get_args()

    save_path = os.path.join(args.output_dir, 'attention_figs')
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(args.device)

    # define the model
    model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large", 
                                            num_seq_knowledge_source=args.num_seq_knowledge_source,
                                            pos3d_encoding=args.pos3d_encoding,
                                            pos3d_depth=args.pos3d_depth,
                                            ).to(device)

    checkpoint.load_checkpoint(args.output_dir, model)

    test_dl = DataLoader(
        dataset=RGBDepthDataset(
                            root_dir=args.test_data_path, 
                            n_frames=2,
                            transform='identity',
                            image_size=args.img_size, 
                            depth_size=args.pos3d_depth, 
                            train_set=False,
                        ),
        batch_size=5,
        num_workers=args.num_workers,
    )


    model.eval()

    calculate_attention_ratio(model=model,
                              test_dl=test_dl,
                              info={
                                  'name': 'noise',
                                  'label': 'Noise Scale',
                                  'severities': [0.0, 0.5, 1.0, 1.5, 2.0],
                                  'fn': lambda x, s: gaussian_noise(x,scale=s),
                              },
                              save_path=save_path)
    
    calculate_attention_ratio(model=model,
                              test_dl=test_dl,
                              info={
                                  'name': 'blur',
                                  'label': 'Blur Sigma',
                                  'severities': [0, 2, 4, 8, 16],
                                  'fn': lambda x, s: gaussian_blur(x,sigma=s),
                              },
                              save_path=save_path)
    
    calculate_attention_ratio(model=model,
                              test_dl=test_dl,
                              info={
                                  'name': 'pixelate',
                                  'label': 'Pixelate Resize',
                                  'severities': [1, 2, 4, 8, 16],
                                  'fn': lambda x, s: pixelate(x,resize=s),
                              },
                              save_path=save_path)
    
    calculate_attention_ratio(model=model,
                              test_dl=test_dl,
                              info={
                                  'name': 'fog',
                                  'label': 'Fog Strength',
                                  'severities': [0, 50 , 100,  150, 200],
                                  'fn': lambda x, y, s: fog_3d(x,depth=y,fog_strength=s),
                              },
                              save_path=save_path)