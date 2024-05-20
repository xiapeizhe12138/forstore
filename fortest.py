import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_coord(shape, ranges=None, flatten=True):

    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

#for window ssm upsize
    x = torch.rand(1,2,5,6)
    B, C, h, w = x.shape
    out = 6
    x_proj = (
                nn.Linear(C+4, out, bias=False),
                nn.Linear(C+4, out, bias=False),
                nn.Linear(C+4, out, bias=False),
                nn.Linear(C+4, out, bias=False),
            )
    x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))  # (K=4, N, inner)
    del x_proj

    coord = make_coord(torch.rand(1,2,round(5*1.2),round(6*1.2)).shape[-2:], flatten=False).unsqueeze(0)
    print(f"coord.shape: {coord.shape}")
    scale = torch.ones(coord.shape)
    scale[:,:,:,0]=2/coord.shape[-2]
    scale[:,:,:,1]=2/coord.shape[-1]
    win = 3
    Lwin = win*win
    pad = int((win-1)/2)
    padding = (pad, pad, pad, pad)

    ref_coord = make_coord(x.shape[-2:])

    B, C, h, w = x.shape
    Bc, H, W, Cc = coord.shape
    Bl = Bc * H * W
    feat_coord = make_coord(x.shape[-2:], flatten=False).permute(2, 0, 1) \
                                .unsqueeze(0).expand(Bc, 2, *x.shape[-2:]).to(coord)             #B,h,w,C

    # Bw = B*H*W
    x = F.pad(x, padding, 'reflect')
    x = F.unfold(x, win).view(B, -1, h, w)
    x = F.grid_sample(x, coord.flip(-1), mode='nearest',                                        #upsize feature
                        align_corners=False).permute(0, 2, 3, 1).contiguous()                   #B,H,W,C
    x = x.view(B*H*W, C, win, win)

    feat_coord = F.pad(feat_coord, padding, 'reflect')
    feat_coord = F.unfold(feat_coord, win).view(Bc, -1, h, w)
    feat_coord = F.grid_sample(feat_coord, coord.flip(-1), mode='nearest',                      #upsize feature coord
                        align_corners=False).permute(0, 2, 3, 1).contiguous()                   #B,H,W,C
    feat_coord = feat_coord.view(Bl, Cc, win, win)

    coord = coord.unsqueeze(-1).expand(Bl, Cc, Lwin).view(Bl, Cc, win, win)#.permute(1, 2, 3, 4, 0).contiguous()   #Bc, Cc, Lwin, H, W
    coord = coord - feat_coord
    del feat_coord
    scale = scale.unsqueeze(-1).expand(Bc, H, W, Cc, Lwin).view(Bl, Cc, win, win)

    x = torch.cat([x, coord, scale], dim=1)
    del coord;del scale
    x = torch.stack([x.view(Bl, -1, Lwin), torch.transpose(x, dim0=-2, dim1=-1).contiguous().view(Bl, -1, Lwin)],
                                dim=1).view(Bl, 2, -1, Lwin)
    x = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)                                         #B*H*W, 4, C+4, Lwin

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", x, x_proj_weight).contiguous()            #B*H*W, 4, Cdct, Lwin
