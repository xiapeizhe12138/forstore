import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import numbers
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref      
import numpy as np

class windowSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            out_channel,
            windowsize=3,
            sumscan=True,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.out_channel = out_channel
        self.d_state = d_state
        self.d_conv = d_conv
        self.sumscan = sumscan
        self.windowsize = windowsize
        self.Lssm = self.windowsize*self.windowsize
        pad = int((self.windowsize-1)/2)
        self.padding = (pad, pad, pad, pad)
        del pad
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # self.ssmdecay = ssmdecay
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # self.conv_y = nn.Conv2d(4*self.d_inner, self.d_inner, 3, stride=1, padding=1)
        self.in_proj = nn.Linear(self.d_model, self.d_inner+self.out_channel, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            **factory_kwargs,)
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        

        self.selective_scan = selective_scan_fn

        self.in_norm = nn.LayerNorm(self.d_model)
        if self.sumscan:
            self.out_norm = nn.LayerNorm(self.d_inner)
            self.addconv = nn.Conv2d(self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        else:
            self.out_norm = nn.LayerNorm(self.out_channel)
            self.addconv = nn.Conv2d(4*self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        self.skipscale = nn.Conv2d(self.d_model, self.out_channel, bias=conv_bias,
                            kernel_size=1, padding=0, **factory_kwargs,)
        self.out_proj = nn.Linear(self.out_channel, self.out_channel, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def getwindows(self, input):
        B, C, H, W = input.shape
        K = 4
        #代表4种扫描方式

        #构造输入的四种序列
        x=F.pad(input, self.padding, 'reflect')
        x = F.unfold(x, self.windowsize)    #B, C*size*size, H*W
        Bs = B*H*W                          # self.Lssm = self.windowsize*self.windowsize
        x = x.permute(0,2,1).contiguous().view(Bs, C, self.windowsize, self.windowsize)
        x = torch.stack([x.view(Bs, -1, self.Lssm), torch.transpose(x, dim0=-2, dim1=-1).contiguous().view(Bs, -1, self.Lssm)],
                             dim=1).view(Bs, 2, -1, self.Lssm)
        x = torch.cat([x, torch.flip(x, dims=[-1])], dim=1) # (Bs, 4, C, L)

        #构造对应的四种BCt
        input = input.view(B, 1, C, H*W)
        x_dbl = torch.cat([input for i in range(K)], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_dbl, self.x_proj_weight).contiguous().view(B*K, -1, H*W)#B, K, Cdct, L
        x_dbl = F.pad(x_dbl.view(B*K, -1, H, W), self.padding, 'reflect')
        x_dbl = F.unfold(x_dbl, self.windowsize).view(B,K,-1,self.Lssm,H*W)        #B, K, Cdbt, size*size, H*W
        x_dbl = x_dbl.permute(0,4,1,2,3).contiguous().view(Bs, K, -1, self.windowsize, self.windowsize)
        x13, x24 = x_dbl.chunk(2, dim=1)
        x_dbl = torch.cat([x13.view(Bs, 2, -1, self.Lssm), torch.transpose(x24, dim0=-2, dim1=-1)
                                                            .contiguous().view(Bs, 2, -1, self.Lssm)], dim=1)
        del x13;del x24     #未反转，                             WH反转
        x_dbl = torch.stack([x_dbl[:,0,:,:], x_dbl[:,2,:,:], torch.flip(x_dbl[:,1,:,:], dims=[-1]), torch.flip(x_dbl[:,3,:,:], dims=[-1])], dim=1) # (Bs, 4, Cdbt, L)
                            #未反转，         WH反转,             首尾反转，                           WH反转且首尾反转，与输入序列反转顺序相同
        return x, x_dbl
        #(Bs, 4, C, L)
    
    def forward_core(self, input: torch.Tensor):
        B, C, H, W = input.shape
        #此时的输入为[B,d_inner,H,W]，可以认为输入在上一步被扩充了channel
        Bl = B*H*W
        K = 4
        #代表4种扫描方式
        #x_proj_weight是x_proj线性层对应的参数，shape为[4,N,inner]，N代表token长度
        #x_proj为将张量从d_inner投到dt_rank + d_state * 2，这是为了生成▲和B,C，
        #这步是为了构造可学习的矩阵，因为B,C,▲是通过输入进行全链接从而生成的。
        xs, x_dbl = self.getwindows(input)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl
        #拆分出dts,Bs,和Cs,这里的S表示S6可学习的ABC▲，此时▲的维度是1代表这一次输入的门控

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(Bl, K, -1, self.Lssm), self.dt_projs_weight)
        #0阶保持求出可学习的dts这时扩充了▲的维度表示多个channel的门控
        xs = xs.float().view(Bl, -1, self.Lssm)#合并多条扫描线路concat
        dts = dts.contiguous().float().view(Bl, -1, self.Lssm) # (b, k * d, l)
        Bs = Bs.float().view(Bl, K, -1, self.Lssm)
        Cs = Cs.float().view(Bl, K, -1, self.Lssm) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)#维度调整
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(Bl, K, -1, self.Lssm)
        del xs;del Bs;del Cs;del dts
        assert out.dtype == torch.float

        #输出按扫描分解
        inv = torch.flip(out[:, 2:4], dims=[-1]).view(Bl, 2, -1, self.Lssm)
        wh = torch.transpose((out[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        out = (out[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)
        invwh = torch.transpose((inv[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        inv = (inv[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)

        if self.sumscan:
            out = out+inv+wh+invwh
            del inv;del wh;del invwh
            out = self.out_norm(out.permute(0, 2, 3, 1).contiguous())
            out = self.addconv(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, self.out_channel)
        else:
            out = torch.cat([out,inv,wh,invwh],dim=1)
            del inv;del wh;del invwh
            out = self.addconv(out).view(B, H, W, self.out_channel)
            out = self.out_norm(out)

        return out

    def forward(self, input: torch.Tensor, outype='hwc',**kwargs):
        x = self.in_norm(input)
        B, H, W, C = x.shape
        #C==d_model
        xz = self.in_proj(x)
        #expand表示维度扩展shortcut，d_Inner=d_model*expand
        #将输入[B,H,W,C]沿着channel括成[B,H,W,2*d_inner]
        x, z = torch.split(xz, [self.d_inner, self.out_channel], dim=-1)
        #将xz拆分为2个[B,H,W,d_inner]
        x = x.permute(0, 3, 1, 2).contiguous()
        #将x改成[B,d_inner,H,W]便于进行conv
        x = self.act(self.conv2d(x))
        #组卷积，一组为d_inner个channel,对每组分别进行
        x = self.forward_core(x)
        x = x * self.act(z)
        
        skip = self.skipscale(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(x) + skip#B, H, W, C
        
        if outype=='chw':
            out = out.permute(0, 3, 1, 2).contiguous()
        return out


###########################################################################################################################################################
###########################################################################################################################################################

class windowSS2Dsimple(nn.Module):
    def __init__(
            self,
            d_model,
            out_channel,
            windowsize=3,
            sumscan=True,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.out_channel = out_channel
        self.d_state = d_state
        self.d_conv = d_conv
        self.sumscan = sumscan
        self.windowsize = windowsize
        self.Lssm = self.windowsize*self.windowsize
        pad = int((self.windowsize-1)/2)
        self.padding = (pad, pad, pad, pad)
        del pad
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # self.ssmdecay = ssmdecay
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # self.conv_y = nn.Conv2d(4*self.d_inner, self.d_inner, 3, stride=1, padding=1)
        self.in_proj = nn.Linear(self.d_model, self.d_inner+self.out_channel, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            **factory_kwargs,)
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        

        self.selective_scan = selective_scan_fn

        self.in_norm = nn.LayerNorm(self.d_model)
        if self.sumscan:
            self.out_norm = nn.LayerNorm(self.d_inner)
            self.addconv=nn.Conv2d(self.d_inner, self.d_inner, kernel_size=self.windowsize, 
                             groups=self.d_inner, stride=1, padding=0, bias=False, **factory_kwargs)
            self.addlinear=nn.Linear(self.d_inner, self.out_channel, bias=False, **factory_kwargs)
            # self.addconv = nn.Conv2d(self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        else:
            self.out_norm = nn.LayerNorm(self.out_channel)
            self.addconv = nn.Conv2d(4*self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        self.skipscale = nn.Conv2d(self.d_model, self.out_channel, bias=conv_bias,
                            kernel_size=1, padding=0, **factory_kwargs,)
        self.out_proj = nn.Linear(self.out_channel, self.out_channel, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def getwindows(self, input):
        B, C, H, W = input.shape
        K = 4
        #代表4种扫描方式

        #构造输入的四种序列
        x=F.pad(input, self.padding, 'reflect')
        x = F.unfold(x, self.windowsize)    #B, C*size*size, H*W
        Bs = B*H*W                          # self.Lssm = self.windowsize*self.windowsize
        x = x.permute(0,2,1).contiguous().view(Bs, C, self.windowsize, self.windowsize)
        x = torch.stack([x.view(Bs, -1, self.Lssm), torch.transpose(x, dim0=-2, dim1=-1).contiguous().view(Bs, -1, self.Lssm)],
                             dim=1).view(Bs, 2, -1, self.Lssm)
        x = torch.cat([x, torch.flip(x, dims=[-1])], dim=1) # (Bs, 4, C, L)

        #构造对应的四种BCt
        input = input.view(B, 1, C, H*W)
        x_dbl = torch.cat([input for i in range(K)], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_dbl, self.x_proj_weight).contiguous().view(B*K, -1, H*W)#B, K, Cdct, L
        x_dbl = F.pad(x_dbl.view(B*K, -1, H, W), self.padding, 'reflect')
        x_dbl = F.unfold(x_dbl, self.windowsize).view(B,K,-1,self.Lssm,H*W)        #B, K, Cdbt, size*size, H*W
        x_dbl = x_dbl.permute(0,4,1,2,3).contiguous().view(Bs, K, -1, self.windowsize, self.windowsize)
        x13, x24 = x_dbl.chunk(2, dim=1)
        x_dbl = torch.cat([x13.view(Bs, 2, -1, self.Lssm), torch.transpose(x24, dim0=-2, dim1=-1)
                                                            .contiguous().view(Bs, 2, -1, self.Lssm)], dim=1)
        del x13;del x24     #未反转，                             WH反转
        x_dbl = torch.stack([x_dbl[:,0,:,:], x_dbl[:,2,:,:], torch.flip(x_dbl[:,1,:,:], dims=[-1]), torch.flip(x_dbl[:,3,:,:], dims=[-1])], dim=1) # (Bs, 4, Cdbt, L)
                            #未反转，         WH反转,             首尾反转，                           WH反转且首尾反转，与输入序列反转顺序相同
        return x, x_dbl
        #(Bs, 4, C, L)
    
    def forward_core(self, input: torch.Tensor):
        B, C, H, W = input.shape
        #此时的输入为[B,d_inner,H,W]，可以认为输入在上一步被扩充了channel
        Bl = B*H*W
        K = 4
        #代表4种扫描方式
        #x_proj_weight是x_proj线性层对应的参数，shape为[4,N,inner]，N代表token长度
        #x_proj为将张量从d_inner投到dt_rank + d_state * 2，这是为了生成▲和B,C，
        #这步是为了构造可学习的矩阵，因为B,C,▲是通过输入进行全链接从而生成的。
        xs, x_dbl = self.getwindows(input)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl
        #拆分出dts,Bs,和Cs,这里的S表示S6可学习的ABC▲，此时▲的维度是1代表这一次输入的门控

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(Bl, K, -1, self.Lssm), self.dt_projs_weight)
        #0阶保持求出可学习的dts这时扩充了▲的维度表示多个channel的门控
        xs = xs.float().view(Bl, -1, self.Lssm)#合并多条扫描线路concat
        dts = dts.contiguous().float().view(Bl, -1, self.Lssm) # (b, k * d, l)
        Bs = Bs.float().view(Bl, K, -1, self.Lssm)
        Cs = Cs.float().view(Bl, K, -1, self.Lssm) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)#维度调整
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(Bl, K, -1, self.Lssm)
        del xs;del Bs;del Cs;del dts
        assert out.dtype == torch.float

        #输出按扫描分解
        inv = torch.flip(out[:, 2:4], dims=[-1]).view(Bl, 2, -1, self.Lssm)
        wh = torch.transpose((out[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        out = (out[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)
        invwh = torch.transpose((inv[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        inv = (inv[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)

        if self.sumscan:
            out = out+inv+wh+invwh
            del inv;del wh;del invwh
            out = self.out_norm(out.permute(0, 2, 3, 1).contiguous())
            out = self.addconv(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, self.out_channel)
            out = self.addlinear(out)
        else:
            out = torch.cat([out,inv,wh,invwh],dim=1)
            del inv;del wh;del invwh
            out = self.addconv(out).view(B, H, W, self.out_channel)
            out = self.out_norm(out)

        return out

    def forward(self, input: torch.Tensor, outype='hwc',**kwargs):
        x = self.in_norm(input)
        B, H, W, C = x.shape
        #C==d_model
        xz = self.in_proj(x)
        #expand表示维度扩展shortcut，d_Inner=d_model*expand
        #将输入[B,H,W,C]沿着channel括成[B,H,W,2*d_inner]
        x, z = torch.split(xz, [self.d_inner, self.out_channel], dim=-1)
        #将xz拆分为2个[B,H,W,d_inner]
        x = x.permute(0, 3, 1, 2).contiguous()
        #将x改成[B,d_inner,H,W]便于进行conv
        x = self.act(self.conv2d(x))
        #组卷积，一组为d_inner个channel,对每组分别进行
        x = self.forward_core(x)
        x = x * self.act(z)
        
        skip = self.skipscale(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(x) + skip#B, H, W, C
        
        if outype=='chw':
            out = out.permute(0, 3, 1, 2).contiguous()
        return out

###########################################################################################################################################################
###########################################################################################################################################################


class multiwindowSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            out_channel,
            windowsize=[3, ],
            channelweight=None,
            sumscan=True,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.out_channel = out_channel
        self.d_state = d_state
        self.d_conv = d_conv
        self.sumscan = sumscan
        self.windowsize = windowsize
        self.step = len(self.windowsize)
        assert self.step<=4 ,"len windowsize too long"
        if channelweight is not None:
            self.channelweight = channelweight
            assert self.step == len(self.channelweight) ,"len windowsize must match len channelweight"
        else:
            self.channelweight = np.ones(self.step)
        # del pad
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            **factory_kwargs,)
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        

        self.selective_scan = selective_scan_fn

        self.in_norm = nn.LayerNorm(self.d_model)
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.convs = []
        self.channel = []
        sum = np.sum(self.channelweight)
        for i in range(self.step):
            outchannel = int(self.out_channel*(self.channelweight[i]/self.step))
            # conv = nn.Conv2d(self.d_inner, outchannel, kernel_size=self.windowsize[i], 
                             # stride=1, padding=0, bias=False, **factory_kwargs)
            if i==0:
                self.conv0=nn.Conv2d(self.d_inner, self.d_inner, kernel_size=self.windowsize[i], 
                             groups=self.d_inner, stride=1, padding=0, bias=False, **factory_kwargs)
                self.linear0=nn.Linear(self.d_inner, outchannel, bias=False, **factory_kwargs)
            elif i==1:
                self.conv1=nn.Conv2d(self.d_inner, self.d_inner, kernel_size=self.windowsize[i], 
                             groups=self.d_inner, stride=1, padding=0, bias=False, **factory_kwargs)
                self.linear1=nn.Linear(self.d_inner, outchannel, bias=False, **factory_kwargs)
            elif i==2:
                self.conv2=nn.Conv2d(self.d_inner, self.d_inner, kernel_size=self.windowsize[i], 
                             groups=self.d_inner, stride=1, padding=0, bias=False, **factory_kwargs)
                self.linear2=nn.Linear(self.d_inner, outchannel, bias=False, **factory_kwargs)
            else:
                assert self.step<=3,"too many window scale"
            # self.convs.append(conv)
            self.channel.append(outchannel)
        self.ssmout = np.sum(self.channel)
        # del conv
        del outchannel

        self.in_proj = nn.Linear(self.d_model, self.d_inner+self.ssmout, bias=bias, **factory_kwargs)
        self.skipscale = nn.Conv2d(self.d_model, self.out_channel, bias=conv_bias,
                            kernel_size=1, padding=0, **factory_kwargs,)
        self.out_proj = nn.Linear(self.ssmout, self.out_channel, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def windowscan(self, input, inx_dbl, windowsize, step):
        B, C, H, W = input.shape
        K = 4
        #代表4种扫描方式
        Lssm = windowsize*windowsize
        pad = int((windowsize-1)/2)
        padding = (pad, pad, pad, pad)
        Bl = B*H*W                          # self.Lssm = self.windowsize*self.windowsize
        #构造输入的四种序列
        xs=F.pad(input, padding, 'reflect')
        xs = F.unfold(xs, windowsize)    #B, C*size*size, H*W
        
        xs = xs.permute(0,2,1).contiguous().view(Bl, C, windowsize, windowsize)
        xs = torch.stack([xs.view(Bl, -1, Lssm), torch.transpose(xs, dim0=-2, dim1=-1).contiguous().view(Bl, -1, Lssm)],
                             dim=1).view(Bl, 2, -1, Lssm)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (Bl, 4, C, L)
        
        x_dbl = F.pad(inx_dbl.view(B*K, -1, H, W), padding, 'reflect')
        x_dbl = F.unfold(x_dbl, windowsize).view(B,K,-1,Lssm,H*W)        #B, K, Cdbt, size*size, H*W
        x_dbl = x_dbl.permute(0,4,1,2,3).contiguous().view(Bl, K, -1, windowsize, windowsize)
        x13, x24 = x_dbl.chunk(2, dim=1)
        x_dbl = torch.cat([x13.view(Bl, 2, -1, Lssm), torch.transpose(x24, dim0=-2, dim1=-1)
                                                            .contiguous().view(Bl, 2, -1, Lssm)], dim=1)
        del x13;del x24     #未反转，                             WH反转
        x_dbl = torch.stack([x_dbl[:,0,:,:], x_dbl[:,2,:,:], torch.flip(x_dbl[:,1,:,:], dims=[-1]), torch.flip(x_dbl[:,3,:,:], dims=[-1])], dim=1) # 
                            #未反转，         WH反转,             首尾反转，                           WH反转且首尾反转，与输入序列反转顺序相同(Bs, 4, Cdbt, L)

        #(Bs, 4, C, L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(Bl, K, -1, Lssm), self.dt_projs_weight)
        #0阶保持求出可学习的dts这时扩充了▲的维度表示多个channel的门控
        xs = xs.float().view(Bl, -1, Lssm)#合并多条扫描线路concat
        dts = dts.contiguous().float().view(Bl, -1, Lssm) # (b, k * d, l)
        Bs = Bs.float().view(Bl, K, -1, Lssm)
        Cs = Cs.float().view(Bl, K, -1, Lssm) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)#维度调整
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
        out = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(Bl, K, -1, Lssm)
        del xs;del Bs;del Cs;del dts;del As;del Ds;del dt_projs_bias
        assert out.dtype == torch.float

        #输出按扫描分解
        inv = torch.flip(out[:, 2:4], dims=[-1]).view(Bl, 2, -1, Lssm)
        wh = torch.transpose((out[:, 1]).view(Bl, -1, windowsize, windowsize), dim0=2, dim1=3).contiguous()
        out = (out[:, 0]).view(Bl, -1, windowsize, windowsize)
        invwh = torch.transpose((inv[:, 1]).view(Bl, -1, windowsize, windowsize), dim0=2, dim1=3).contiguous()
        inv = (inv[:, 0]).view(Bl, -1, windowsize, windowsize)

        out = out+inv+wh+invwh
        del inv;del wh;del invwh
        out = self.out_norm(out.permute(0, 2, 3, 1).contiguous())
        if step==0:
            out = self.conv0(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, -1)
            out = self.linear0(out)
        elif step==1:
            out = self.conv1(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, -1)
            out = self.linear1(out)
        elif step==2:
            out = self.conv2(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, -1)
            out = self.linear2(out)
            
        return out
        
        
    
    def forward_core(self, input: torch.Tensor):
        B, C, H, W = input.shape
        #此时的输入为[B,d_inner,H,W]，可以认为输入在上一步被扩充了channel
        # Bl = B*H*W
        K = 4
        #代表4种扫描方式
        #构造对应的四种BCt
        x_dbl = input.view(B, 1, C, H*W)
        x_dbl = torch.cat([x_dbl for i in range(K)], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_dbl, self.x_proj_weight).contiguous().view(B*K, -1, H*W)#B, K, Cdct, L
        #x_proj_weight是x_proj线性层对应的参数，shape为[4,N,inner]，N代表token长度
        output = []
        for i in range(self.step):
            out = self.windowscan(input, x_dbl, self.windowsize[i], i)
            output.append(out)
            del out
        del x_dbl
        output = torch.cat(output, dim=-1)
        
        #x_proj为将张量从d_inner投到dt_rank + d_state * 2，这是为了生成▲和B,C，
        #这步是为了构造可学习的矩阵，因为B,C,▲是通过输入进行全链接从而生成的。

        return output

    def forward(self, input: torch.Tensor, outype='hwc',**kwargs):
        x = self.in_norm(input)
        B, H, W, C = x.shape
        #C==d_model
        xz = self.in_proj(x)
        #expand表示维度扩展shortcut，d_Inner=d_model*expand
        #将输入[B,H,W,C]沿着channel括成[B,H,W,2*d_inner]
        x, z = torch.split(xz, [self.d_inner, self.ssmout], dim=-1)
        del xz
        #将xz拆分为2个[B,H,W,d_inner]
        x = x.permute(0, 3, 1, 2).contiguous()
        #将x改成[B,d_inner,H,W]便于进行conv
        x = self.act(self.conv2d(x))
        #组卷积，一组为d_inner个channel,对每组分别进行
        x = self.forward_core(x)
        x = x * self.act(z)
        
        skip = self.skipscale(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(x) + skip#B, H, W, C
        
        if outype=='chw':
            out = out.permute(0, 3, 1, 2).contiguous()
        return out

###########################################################################################################################################################
###########################################################################################################################################################

class windowSS2D3(nn.Module):
    def __init__(
            self,
            d_model,
            out_channel,
            windowsize=3,
            sumscan=True,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.out_channel = out_channel
        self.d_state = d_state
        self.d_conv = d_conv
        self.sumscan = sumscan
        self.windowsize = windowsize
        self.Lssm = self.windowsize*self.windowsize
        pad = int((self.windowsize-1)/2)
        self.padding = (pad, pad, pad, pad)
        del pad
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # self.ssmdecay = ssmdecay
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # self.conv_y = nn.Conv2d(4*self.d_inner, self.d_inner, 3, stride=1, padding=1)
        self.in_proj = nn.Linear(self.d_model, self.d_inner+self.out_channel, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            **factory_kwargs,)
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        

        self.selective_scan = selective_scan_fn

        self.in_norm = nn.LayerNorm(self.d_model)
        if self.sumscan:
            self.out_norm = nn.LayerNorm(self.d_inner)
            self.addconv = nn.Conv2d(self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        else:
            self.out_norm = nn.LayerNorm(self.out_channel)
            self.addconv = nn.Conv2d(4*self.d_inner, self.out_channel, self.windowsize, stride=1, padding=0)
        self.skipscale = nn.Conv2d(self.d_model, self.out_channel, bias=conv_bias,
                            kernel_size=3, padding=1, **factory_kwargs,)
        self.out_proj = nn.Linear(self.out_channel, self.out_channel, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def getwindows(self, input):
        B, C, H, W = input.shape
        K = 4
        #代表4种扫描方式

        #构造输入的四种序列
        x=F.pad(input, self.padding, 'reflect')
        x = F.unfold(x, self.windowsize)    #B, C*size*size, H*W
        Bs = B*H*W                          # self.Lssm = self.windowsize*self.windowsize
        x = x.permute(0,2,1).contiguous().view(Bs, C, self.windowsize, self.windowsize)
        x = torch.stack([x.view(Bs, -1, self.Lssm), torch.transpose(x, dim0=-2, dim1=-1).contiguous().view(Bs, -1, self.Lssm)],
                             dim=1).view(Bs, 2, -1, self.Lssm)
        x = torch.cat([x, torch.flip(x, dims=[-1])], dim=1) # (Bs, 4, C, L)

        #构造对应的四种BCt
        input = input.view(B, 1, C, H*W)
        x_dbl = torch.cat([input for i in range(K)], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_dbl, self.x_proj_weight).contiguous().view(B*K, -1, H*W)#B, K, Cdct, L
        x_dbl = F.pad(x_dbl.view(B*K, -1, H, W), self.padding, 'reflect')
        x_dbl = F.unfold(x_dbl, self.windowsize).view(B,K,-1,self.Lssm,H*W)        #B, K, Cdbt, size*size, H*W
        x_dbl = x_dbl.permute(0,4,1,2,3).contiguous().view(Bs, K, -1, self.windowsize, self.windowsize)
        x13, x24 = x_dbl.chunk(2, dim=1)
        x_dbl = torch.cat([x13.view(Bs, 2, -1, self.Lssm), torch.transpose(x24, dim0=-2, dim1=-1)
                                                            .contiguous().view(Bs, 2, -1, self.Lssm)], dim=1)
        del x13;del x24     #未反转，                             WH反转
        x_dbl = torch.stack([x_dbl[:,0,:,:], x_dbl[:,2,:,:], torch.flip(x_dbl[:,1,:,:], dims=[-1]), torch.flip(x_dbl[:,3,:,:], dims=[-1])], dim=1) # (Bs, 4, Cdbt, L)
                            #未反转，         WH反转,             首尾反转，                           WH反转且首尾反转，与输入序列反转顺序相同
        return x, x_dbl
        #(Bs, 4, C, L)
    
    def forward_core(self, input: torch.Tensor):
        B, C, H, W = input.shape
        #此时的输入为[B,d_inner,H,W]，可以认为输入在上一步被扩充了channel
        Bl = B*H*W
        K = 4
        #代表4种扫描方式
        #x_proj_weight是x_proj线性层对应的参数，shape为[4,N,inner]，N代表token长度
        #x_proj为将张量从d_inner投到dt_rank + d_state * 2，这是为了生成▲和B,C，
        #这步是为了构造可学习的矩阵，因为B,C,▲是通过输入进行全链接从而生成的。
        xs, x_dbl = self.getwindows(input)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl
        #拆分出dts,Bs,和Cs,这里的S表示S6可学习的ABC▲，此时▲的维度是1代表这一次输入的门控

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(Bl, K, -1, self.Lssm), self.dt_projs_weight)
        #0阶保持求出可学习的dts这时扩充了▲的维度表示多个channel的门控
        xs = xs.float().view(Bl, -1, self.Lssm)#合并多条扫描线路concat
        dts = dts.contiguous().float().view(Bl, -1, self.Lssm) # (b, k * d, l)
        Bs = Bs.float().view(Bl, K, -1, self.Lssm)
        Cs = Cs.float().view(Bl, K, -1, self.Lssm) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)#维度调整
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(Bl, K, -1, self.Lssm)
        del xs;del Bs;del Cs;del dts
        assert out.dtype == torch.float

        #输出按扫描分解
        inv = torch.flip(out[:, 2:4], dims=[-1]).view(Bl, 2, -1, self.Lssm)
        wh = torch.transpose((out[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        out = (out[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)
        invwh = torch.transpose((inv[:, 1]).view(Bl, -1, self.windowsize, self.windowsize), dim0=2, dim1=3).contiguous()
        inv = (inv[:, 0]).view(Bl, -1, self.windowsize, self.windowsize)

        if self.sumscan:
            out = out+inv+wh+invwh
            del inv;del wh;del invwh
            out = self.out_norm(out.permute(0, 2, 3, 1).contiguous())
            out = self.addconv(out.permute(0, 3, 1, 2).contiguous()).view(B, H, W, self.out_channel)
        else:
            out = torch.cat([out,inv,wh,invwh],dim=1)
            del inv;del wh;del invwh
            out = self.addconv(out).view(B, H, W, self.out_channel)
            out = self.out_norm(out)

        return out

    def forward(self, input: torch.Tensor, outype='hwc',**kwargs):
        x = self.in_norm(input)
        B, H, W, C = x.shape
        #C==d_model
        xz = self.in_proj(x)
        #expand表示维度扩展shortcut，d_Inner=d_model*expand
        #将输入[B,H,W,C]沿着channel括成[B,H,W,2*d_inner]
        x, z = torch.split(xz, [self.d_inner, self.out_channel], dim=-1)
        #将xz拆分为2个[B,H,W,d_inner]
        x = x.permute(0, 3, 1, 2).contiguous()
        #将x改成[B,d_inner,H,W]便于进行conv
        x = self.act(self.conv2d(x))
        #组卷积，一组为d_inner个channel,对每组分别进行
        x = self.forward_core(x)
        x = x * self.act(z)
        
        skip = self.skipscale(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(x) + skip#B, H, W, C
        
        if outype=='chw':
            out = out.permute(0, 3, 1, 2).contiguous()
        return out