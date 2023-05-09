import glob
import os
from pathlib import Path

import torch


def save_checkpoint(path, epoch, model, optimizer):
    path = Path(path)
    
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    torch.save(state_dict, path / ('checkpoint-%s.pth' % epoch))


def load_checkpoint(path, model, optimizer):
    path = Path(path)
    all_checkpoints = glob.glob(os.path.join(path, 'checkpoint-*.pth'))
    sorted_ckpt = []
    
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        if t.isdigit():
            sorted_ckpt.append(int(t))
    sorted_ckpt = sorted(sorted_ckpt)

    if len(sorted_ckpt) == 0:
        return 0

    path = os.path.join(path, 'checkpoint-%d.pth' % sorted_ckpt[-1])

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch']
    

def iter_checkpoints(args, model_without_ddp, last_checkpoint=False):
    import glob
    output_dir = Path(args.output_dir)
    all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
    sorted_ckpt = []
    
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        if t.isdigit():
            sorted_ckpt.append(int(t))
    sorted_ckpt = sorted(sorted_ckpt)
    if last_checkpoint:
        sorted_ckpt = [sorted_ckpt[-1]]

    for ckpt_num in sorted_ckpt:
        path = os.path.join(output_dir, 'checkpoint-%d.pth' % ckpt_num)

        checkpoint = torch.load(path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            yield checkpoint['epoch']
        else:
            yield ckpt_num

