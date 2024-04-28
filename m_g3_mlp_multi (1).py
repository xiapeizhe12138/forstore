import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init
from mmedit.utils import get_root_logger
from mmedit.datasets.pipelines.utils import make_coord
from mmedit.models.builder import build_backbone, build_component
from mmedited.models.common.arch_csnln import CrossScaleAttention
from einops import repeat
import math
import numpy as np
import time
import pdb
from mmedited.models.backbones.change_ssm.windowss2d import multiwindowSS2D
from mmedited.models.backbones.sr_backbones.ss2d import SS2D

class MgobalSRNet(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 SSM,
                 LSSM,
                 imnet,
                 featcat=False,
                 cropstep=128,
                 mincrop=64,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local=True,
                 softmax_scale=1,
                #  unfold_channel=3,
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.non_local = non_local
        self.featcat = featcat
        # self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale
        self.cropstep = cropstep
        self.mincrop = mincrop
        # self.unfold_channel = unfold_channel
        
        self.encoder = build_backbone(encoder)
        if hasattr(self.encoder, 'mid_channels'):
            imnet_dim = self.encoder.mid_channels
        else:
            imnet_dim = self.encoder.embed_dim
        
        # self.upchannel = imnet_dim * self.unfold_channel
        
        if self.feat_unfold:
            LSSM['d_model'] = imnet_dim
            SSM['in_dim'] = LSSM['out_channel_multi']*imnet_dim
        else:
            SSM['in_dim'] = imnet_dim
        
        if self.non_local:
            SSM['in_dim'] += imnet_dim
            

        SSM['in_dim'] += 4
        SSM['out_channel'] = int(SSM['in_dim']/4)
        imnet['in_dim'] = SSM['out_channel']*self.local_size*self.local_size
        
        self.global_SS2D = SS2D(d_model=SSM['in_dim'],
                    out_channel=SSM['out_channel'],
                    d_state=SSM['d_state'],
                    d_conv=SSM['d_conv'],
                    expand=SSM['expand']) 
        self.imnet = build_component(imnet) 
        
        if self.non_local:
            self.non_local_ssm = SS2D(d_model=imnet_dim,
                            out_channel=imnet_dim)    
        self.localssm = multiwindowSS2D(d_model=LSSM['d_model'],
                            out_channel=LSSM['out_channel_multi']*imnet_dim,
                            windowsize=LSSM['windowsize'],
                            channelweight=LSSM['channelweight'],
                            sumscan=LSSM['sumscan'],
                            # windowsize=LSSM['windowsize'],
                            expand=LSSM['expand'])
            
        self.countgenfeat = False
            

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        # print(x.shape)
        # print(coord.shape)
        B, C, h, w = x.shape
        Bc, H, W, _ = coord.shape
        feature = self.gen_feature(x)

        if self.countgenfeat:
            
            return feature
        else:
            if self.eval_bsize is None or not test_mode:
                pred = self.query_rgb(feature, coord, cell)
            else:
                pred = self.batched_predict(feature, coord, cell)

            pred += F.grid_sample(x, coord.view(Bc,H*W,-1).flip(-1).unsqueeze(1), mode='bilinear',\
                        padding_mode='border', align_corners=False)[:, :, 0, :].view(B,-1,H,W)#

            return pred


    def query_rgb(self, features, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res = []
        # print(features[1])  error      features is just a tensor in dict  
        for feature in features:
            
            bs, h, w, _ = coord.shape
            Br, Hr, Wr, Cr = scale.shape
            assert (h==Hr and w==Wr) , "size of coord and cell not match"

            if self.feat_unfold:
                featureC = feature.permute(0, 2, 3, 1).contiguous()
                feat = self.localssm(featureC, outype='chw')
                if self.non_local:
                    non_local_feat = self.non_local_ssm(featureC, outype='chw')#mamba
                    feat = torch.cat([feat, non_local_feat], dim=1)
                    del non_local_feat
                del featureC
                if self.featcat:
                    feat = torch.cat([feature, feat], dim=1)
            else:
                feat = feature
            
            B, C, H, W = feat.shape
            #low B, C, H, W
            #high B, C, h, w

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])                   #B,C,h,w
            feat_coord = feat_coord.to(coord)
            del feature
            
            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            
            preds = []
            areas = []
            
            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:,0,0,0])).view(B,  1)
                ty = ((W - 1) / (1 - scale[:,0,0,1])).view(B,  1)
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0
                
                coord_ = coord.clone().view(bs, h*w, -1)
                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift
                coord_ = coord_.clamp_(-1 + 1e-6, 1 - 1e-6).view(bs, h, w, 2)
                
                # value
                upfeat = F.grid_sample(feat, coord_.flip(-1), mode='nearest',                 #upsize feature
                    align_corners=False).permute(0, 2, 3, 1).contiguous()                     #B,C,h,w

                #Interpolate feat_coord to HR resolution
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1),                          #upsize feature coord
                    mode='nearest', align_corners=False).permute(0, 2, 3, 1).contiguous()     #B,C,h,w

                rel_coord = coord - coord_k                                                   #B,C,h,w
                rel_coord[:, :, :, 0] *= (H/2)                                    #relative coord information
                rel_coord[:, :, :, 1] *= (W/2)
                del coord_;del coord_k

                scale_ = scale.clone()                                                        #B,C,h,w
                scale_[:, :, :, 0] *= (H/2)                                       #size information
                scale_[:, :, :, 1] *= (W/2)

                inp = torch.cat([upfeat, rel_coord, scale_], dim=-1).contiguous()             #input get
                del scale_;del upfeat;del rel_coord

                pred = self.global_SS2D(inp, outype='hwc')                                           #B,h,w,C
                del inp

                preds.append(pred)
                del pred

            ret = torch.cat(preds, dim=-1).view(bs, h*w, -1)
            
            ret = self.imnet(ret).view(bs, h, w, -1)
            res.append(ret)
            del ret
    
        result = torch.cat(res, dim=-1).permute(0, 3, 1, 2).contiguous()                       #B,C,h,w
        B, C, H, W=result.shape
        assert (B==bs and C==3) and (H==h and W==w),'wrong output size'
        return result

    def cropdim(self, h):
        #我们希望将step转换成一个起始元素0，结束元素lenth/-1的递增序列，相邻元素之间间隔为crop，最末间隔不低于mincrop的数组
        step = np.arange(0, h, self.cropstep)
        if(h-step[-1] < self.mincrop):
            step[-1] = h
        else:
            step = np.append(step,h)
        return step
    
    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad(): 
            B, C, h, w = x[0].shape 
            assert B==1, "batchsize not match"
            Bc, Hc, Wc, _ = coord.shape
            Br, Hr, Wr, _ = cell.shape
            H = round(float(2/cell[0,0,0,0]))
            W = round(float(2/cell[0,0,0,1]))
            assert H==Hc and H==Hr, "inputsize not match"
            assert W==Wc and W==Wr, "inputsize not match"
            # print(f"output.shape  {coord.shape}")
            hspace = self.cropdim(H)#, crop=120, mincrop=60)
            wspace = self.cropdim(W)#, crop=120, mincrop= 60)
            preds_hcrop = []
            for i in range(len(hspace)-1):
                preds_hwcrop = []
                for j in range(len(wspace)-1):
                    pred = self.query_rgb(x, coord[:, hspace[i]:hspace[i+1], wspace[j]:wspace[j+1], :].contiguous(),
                                      cell[:, hspace[i]:hspace[i+1], wspace[j]:wspace[j+1], :].contiguous())
                    preds_hwcrop.append(pred)
                pred = torch.cat(preds_hwcrop, dim=3)
                # print(f"preds_hcrop.shape  {pred.shape}")
                preds_hcrop.append(pred)
            pred = torch.cat(preds_hcrop, dim=2)
        return pred

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class MgobalImplicitSRRDN(MgobalSRNet):
    """ITSRN net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 SSM,
                 LSSM,
                 imnet,
                 featcat=False,
                 cropstep=128,
                 mincrop=64,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local=True,
                #  multi_scale=[2],
                 softmax_scale=1,
                #  unfold_channel=3,
                 ):
        super().__init__(
            encoder=encoder,
            SSM=SSM,
            LSSM=LSSM,
            imnet=imnet,
            featcat=featcat,
            cropstep=cropstep,
            mincrop=mincrop,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            non_local=non_local,
            # multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            # unfold_channel=unfold_channel,
            )

        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_blocks = self.encoder.num_blocks
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        return [x]


class LocalImplicitSREDSR(MgobalSRNet):
    """LocalImplicitSR based on EDSR.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_q=imnet_q,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            non_local_attn=non_local_attn,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )

        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x

        return [res]


class LocalImplicitSRSWINIR(MgobalSRNet):
    """ITSRN net based on EDSR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 window_size,
                 encoder,
                 imnet_q,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_q=imnet_q,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            non_local_attn=non_local_attn,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )

        self.window_size = window_size

        self.conv_first = self.encoder.conv_first

        # body
        self.patch_embed = self.encoder.patch_embed
        self.pos_drop = self.encoder.pos_drop
        self.layers = self.encoder.layers
        self.norm = self.encoder.norm
        self.patch_unembed = self.encoder.patch_unembed

        self.conv_after_body = self.encoder.conv_after_body

        # self.apply(self._init_weights)

        del self.encoder

    def forward_features(self, x):

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gen_feature(self, img):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % self.window_size != 0:
            mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            mod_pad_w = self.window_size - w % self.window_size
        img_pad = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        x = self.conv_first(img_pad)
        res = self.forward_features(x)
        res = self.conv_after_body(res)
        res += x

        _, _, h, w = res.size()
        res = res[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]

        return [res]



class LocalImplicitSRNetRGBout(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_qr,
                 imnet_qg,
                 imnet_qb,
                 imnet_k,
                 imnet_v,
                 query_mlp,
                 key_mlp,
                 value_mlp,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale

        # imnet
        self.encoder = build_backbone(encoder)
        if hasattr(self.encoder, 'mid_channels'):
            imnet_dim = self.encoder.mid_channels
        else:
            imnet_dim = self.encoder.embed_dim
        if self.feat_unfold:
            imnet_qr['in_dim'] = imnet_dim * 9
            imnet_qg['in_dim'] = imnet_dim * 9
            imnet_qb['in_dim'] = imnet_dim * 9
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim * 9
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim * 9
        else:
            imnet_qr['in_dim'] = imnet_dim
            imnet_qg['in_dim'] = imnet_dim
            imnet_qb['in_dim'] = imnet_dim
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim

        imnet_k['in_dim'] += 4
        imnet_v['in_dim'] += 4
        
        if self.non_local_attn:
            imnet_qr['in_dim'] += imnet_dim*len(multi_scale)
            imnet_qg['in_dim'] += imnet_dim*len(multi_scale)
            imnet_qb['in_dim'] += imnet_dim*len(multi_scale)
            imnet_v['in_dim'] += imnet_dim*len(multi_scale)
            imnet_v['out_dim'] += imnet_dim*len(multi_scale)

        # imnet_q['in_dim'] *= 2

        self.imnet_qr = build_component(imnet_qr) 
        self.imnet_qg = build_component(imnet_qg)
        self.imnet_qb = build_component(imnet_qb)
        self.imnet_k = build_component(imnet_k) 
        self.imnet_v = build_component(imnet_v) 
        
        if self.non_local_attn:
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)    
            

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        feature = self.gen_feature(x)

        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feature, coord, cell)
        else:
            pred = self.batched_predict(feature, coord, cell)

        pred += F.grid_sample(x, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                    padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return pred


    def query_rgb(self, features, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res_features = []
        for feature in features:
            
            B, C, H, W = feature.shape      #[16, 64, 48, 48]

            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                feat_k = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                if self.non_local_attn:
                    non_local_feat_v = self.cs_attn(feature)                        #[16, 64, 48, 48]
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
                    feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)           #[16, 576+64, 48, 48]
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
            else:
                feat_q = feat_k = feat_v = feature


            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False).permute(0, 3, 2, 1).contiguous()       #[16, 2304, 1, 576]

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])         #[16, 2, 48, 48]
            feat_coord = feat_coord.to(coord)

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []
            
            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:,0,0])).view(B,  1)     # [16, 1]
                ty = ((W - 1) / (1 - scale[:,0,1])).view(B,  1)     # [16, 1]
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0          # [16, 1]
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0          # [16, 1]
                
                bs, q = coord.shape[:2]     
                coord_ = coord.clone()  # [16, 2304, 2]
                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift  # [16, 2304]
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 576]
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 576]

                #Interpolate K to HR resolution
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)       #[16, 2304, 2]

                Q, K = coord, coord_k   #[16, 2304, 2]
                rel = Q - K             #[16, 2304, 2]
                rel[:, :, 0] *= feature.shape[-2]   # without mul
                rel[:, :, 1] *= feature.shape[-1]
                inp = rel   #[16, 2304, 2]

                scale_ = scale.clone()      #[16, 2304, 2]
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                inp_v = torch.cat([value, inp, scale_], dim=-1)   #[16, 2304, 580]
                inp_k = torch.cat([key, inp, scale_], dim=-1)     #[16, 2304, 580]
                    

                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)

                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                pred_k = (key * weight_k).view(bs, q, -1)                     #[16, 2304, 576]
                
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                pred_v = (value * weight_v).view(bs, q, -1)                   #[16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)

            preds_k = torch.stack(preds_k, dim=-1)                      # [16, 2304, 576, 4]
            preds_v = torch.stack(preds_v, dim=-2)                      # [16, 2304, 4, 576]

            attn = (query @ preds_k)                                    # [16, 2304, 1, 4]
            x = ((attn/self.softmax_scale).softmax(dim=-1) @ preds_v)   # [16, 2304, 1, 576]
            x = x.view(bs*q, -1)                                        # [16*2304, 576]

            res_features.append(x) 

        result = torch.cat(res_features, dim=-1)    # [16, 2304, 576x2]
        resultr = self.imnet_qr(result)               # [16, 2304, 1]
        resultg = self.imnet_qg(result)
        resultb = self.imnet_qb(result)
        result = torch.cat((resultr, resultg, resultb), dim=-1).view(bs, q, -1)

        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad(): 
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :],
                                      cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class LocalImplicitSRRDNrgb(LocalImplicitSRNetRGBout):
    """ITSRN net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder,
                 imnet_qr,
                 imnet_qg,
                 imnet_qb,
                 imnet_k,
                 imnet_v,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder=encoder,
            imnet_qr=imnet_qr,
            imnet_qg=imnet_qg,
            imnet_qb=imnet_qb,
            imnet_k=imnet_k,
            imnet_v=imnet_v,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            non_local_attn=non_local_attn,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )


        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_blocks = self.encoder.num_blocks
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        return [x]

if __name__ == "__main__":
    encoder=dict(
            type='RDN',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
    imnet_q=dict(
        type='MLPRefiner',
        in_dim=4,
        out_dim=3,
        hidden_list=[256, 256, 256, 256]),
    imnet_k=dict(
        type='MLPRefiner',
        in_dim=64,
        out_dim=64,
        hidden_list=[256, 256, 256, 256]),
    imnet_v=dict(
        type='MLPRefiner',
        in_dim=64,
        out_dim=64,
        hidden_list=[256, 256, 256, 256])

    # model1 = LocalImplicitSRRDN(encoder, imnet_q, imnet_k, imnet_v)
    # model1.load_state_dict(torch.load('rdn.pth'))
    # model1.eval()