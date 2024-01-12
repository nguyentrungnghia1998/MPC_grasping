import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# Call Mish activation function


# Create a BaseModule class

class BaseModule(nn.Module):
    class BaseModule(torch.nn.Module):
        def __init__(self):
            super(BaseModule, self).__init__()

        @property
        def nparams(self):
            """
            Returns number of trainable parameters of the module.
            """
            num_params = 0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    num_params += np.prod(param.detach().cpu().numpy().shape)
            return num_params


        def relocate_input(self, x: list):
            """
            Relocates provided tensors to the same device set for the module.
            """
            device = next(self.parameters()).device
            for i in range(len(x)):
                if isinstance(x[i], torch.Tensor) and x[i].device != device:
                    x[i] = x[i].to(device)
            return x
        

class Mish(BaseModule):

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
    
class Rezero(BaseModule):

    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Residual(BaseModule):

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output
        
class SeparableConv2d(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
        
class SeparableBlock(BaseModule):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            SeparableConv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


# Create an Separable Resnet Block

class SeparableResnetBlock(BaseModule):

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, dim_out)

        self.block1 = SeparableBlock(dim, dim_out, groups=groups)
        self.block2 = SeparableBlock(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


# Create a Linear Attention Block
    
class SeparableLinearAttention(BaseModule):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_q = SeparableConv2d(dim, self.hidden_dim, 1, 1, 0, 1, False)
        self.to_k = SeparableConv2d(dim, self.hidden_dim, 1, 1, 0, 1, False)
        self.to_v = SeparableConv2d(dim, self.hidden_dim, 1, 1, 0, 1, False)
        self.to_out = SeparableConv2d(self.hidden_dim, dim, 1, 1, 0, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.to_q(x).reshape((b, self.heads, -1, h * w))  # (b,heads,d,h*w)
        k = self.to_k(x).reshape((b, self.heads, -1, h * w))  # (b,heads,d,h*w)
        v = self.to_v(x).reshape((b, self.heads, -1, h * w))  # (b,heads,e,h*w)
        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.permute(0, 1, 3, 2))  # (b,heads,d,e)
        out = torch.matmul(context.permute(0, 1, 3, 2), q)  # (b,heads,e,n)
        out = out.reshape(b, self.hidden_dim, h, w)
        return self.to_out(out)

# Create an Unet receive input Nx4x224x224 and output Nx4x224x224, using lightweight unet in LightGrad
    
class Downsample(BaseModule):

    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample(BaseModule):

    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
class Block(BaseModule):

    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask
    
class SinusoidalPosEmb(BaseModule):

    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class LightGrad_Unet(nn.Module):
    def __init__(self, num_classes=4, dims = [4, 64, 128, 256], time_dim = 128):
        super().__init__()
        
        self.num_classes = num_classes
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.pe_scale = 1000
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.time_pos_emb = SinusoidalPosEmb(time_dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(time_dim, time_dim * 4), Mish(),
                                       torch.nn.Linear(time_dim * 4, time_dim), Mish())

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList([
                    SeparableResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                    SeparableResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(Rezero(SeparableLinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else torch.nn.Identity()
                ]))
            
        mid_dim = dims[-1]  # 256
        self.mid_block1 = SeparableResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(Rezero(SeparableLinearAttention(mid_dim)))
        self.mid_block2 = SeparableResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList([
                    SeparableResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    SeparableResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(Rezero(SeparableLinearAttention(dim_in))),
                    Upsample(dim_in)
                ]))
            
        self.final_block = Block(dims[1], dims[1])
        self.final_conv = torch.nn.Conv2d(dims[1], num_classes, 1)

    def forward(self, x, t, img_feat):
        # x: Nx4x224x224
        # output: Nx4x224x224
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)
        t = t + img_feat
        hiddens = []
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)


        mask_mid = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        mask_up = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        x = self.final_block(x, mask_up)
        x = self.final_conv(x * mask_up)
        return x