'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z, scale=None, start_scale=None):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        if scale is None:
            d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
                2 * torch.matmul(z_flattened, self.embedding.weight.t())
        elif start_scale is None:
            num = int(scale * self.codebook_size)
            d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight[:num, ...]**2).sum(1) - \
                2 * torch.matmul(z_flattened, self.embedding.weight[:num, ...].t())
        else:
            num = int(scale * self.codebook_size)
            num_start = int(start_scale * self.codebook_size)
            d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight[num_start:num, ...]**2).sum(1) - \
                2 * torch.matmul(z_flattened, self.embedding.weight[num_start:num, ...].t())

        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        # min_encoding_scores = torch.exp(-min_encoding_scores/10)

        if scale is None:
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        elif start_scale is None:
            min_encodings = torch.zeros(min_encoding_indices.shape[0], num).to(z)
        else:
            min_encodings = torch.zeros(min_encoding_indices.shape[0], num-num_start).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        if scale is None:
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        elif start_scale is None:
            z_q = torch.matmul(min_encodings, self.embedding.weight[:num, ...]).view(z.shape)
        else:
            z_q = torch.matmul(min_encodings, self.embedding.weight[num_start:num, ...]).view(z.shape)
        # compute loss for embedding
        #loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)


        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
            }

    def get_warp_codebook_loss(self, driving_feat, quant_warp_feat):
        loss = self.beta * torch.mean((quant_warp_feat.detach()-driving_feat)**2) + torch.mean((quant_warp_feat - driving_feat.detach()) ** 2)
        return loss

    def get_codebook_feat(self, indices, shape):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k) 
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1) 
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf 
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size 
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)
   

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x

class MSGenerator(nn.Module):
    def __init__(self, nf=64, emb_dim=256, ch_mult=[1, 2, 2, 4], res_blocks=2, # res_blocks: only used for last resolution block
                img_size=256, attn_resolutions=[32]):
        super().__init__()
        self.nf = nf 
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size 
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        #curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        assert self.num_resolutions == 4
        
        blocks_0 = []
        blocks_1 = []
        blocks_2 = []
        blocks_3 = []
        blocks = []

        for i in range(self.num_resolutions): # i=0: 32; i=1: 64; 2: 128; 3: 256
            curr_res = self.resolution // 2 ** (self.num_resolutions-i-1)
            if i != self.num_resolutions-1:
                eval('blocks_'+str(i)).append(nn.Conv2d(self.nf * self.ch_mult[-1-i], self.nf * self.ch_mult[-1-i], kernel_size=3, stride=1, padding=1))
            if i == 0:
                #eval('blocks_'+str(i)).append()
                eval('blocks_'+str(i)).append(ResBlock(block_in_ch, block_in_ch))
                eval('blocks_'+str(i)).append(AttnBlock(block_in_ch))
                eval('blocks_'+str(i)).append(ResBlock(block_in_ch, block_in_ch))
                
            block_out_ch = self.nf * self.ch_mult[-1-i]
            block_in_ch = self.nf * self.ch_mult[-1-i]

            for _ in range(self.num_resolutions-1-i):
                eval('blocks_'+str(i)).append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    eval('blocks_'+str(i)).append(AttnBlock(block_in_ch))

            if i != self.num_resolutions-1:
                for _ in range(self.num_resolutions-1-i):
                    eval('blocks_'+str(i)).append(Upsample(block_in_ch))
            
        
        block_in_ch = 0
        for mul in ch_mult:
            block_in_ch += mul
        block_in_ch *= nf
        for _ in range(res_blocks):
            blocks.append(ResBlock(block_in_ch, block_out_ch))
            block_in_ch = block_out_ch
            if curr_res in self.attn_resolutions:
                blocks.append(AttnBlock(block_in_ch))
                
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))


        
        '''#blocks = []
        # initial conv
        #blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))'''

        #self.blocks = nn.ModuleList(blocks)
        self.blocks_0 = nn.ModuleList(blocks_0)
        self.blocks_1 = nn.ModuleList(blocks_1)
        self.blocks_2 = nn.ModuleList(blocks_2)
        self.blocks_3 = nn.ModuleList(blocks_3)
        self.blocks = nn.ModuleList(blocks)
   

    def forward(self, x0, x1, x2, x3):

        for block in self.blocks_0:
            x0 = block(x0)
        for block in self.blocks_1:
            x1 = block(x1)
        for block in self.blocks_2:
            x2 = block(x2)
        for block in self.blocks_3:
            x3 = block(x3)
        
        x = torch.cat((x0,x1,x2,x3), dim=1)
        for block in self.blocks:
            x = block(x)
            
        return x

  
@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = Generator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')


    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats



# patch based discriminator
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        return self.main(x)


@ARCH_REGISTRY.register()
class AutoEncoder(nn.Module):
    # removed codebook from VQAutoEncoder
    def __init__(self, img_size, nf, ch_mult, res_blocks=2, attn_resolutions=[16], emb_dim=256, model_path=None):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )

        self.generator = Generator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'ae is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'ae is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')


    def forward(self, x):
        x = self.encoder(x)
        #quant, codebook_loss, quant_stats = self.quantize(x)
        #x = self.generator(quant)
        x = self.generator(x)
        return x #, codebook_loss, quant_stats

@ARCH_REGISTRY.register()
class AutoEncoder_MSDecoder(nn.Module):
    # MSGenerator for AutoEncoder
    def __init__(self, img_size, nf, ch_mult, res_blocks=2, attn_resolutions=[16], emb_dim=256, model_path=None, connect_list = []):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )

        self.generator = MSGenerator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions
        )

        self.fuse_encoder_block = {'256':2, '128':5, '64':8, '32':11}

        self.connect_list = connect_list

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'ae is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'ae is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')


    def forward(self, x):
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        
        #enc_feat_dict[str(x.shape[-1])] = x.clone()
        #x = self.encoder(x)
        if '64' in enc_feat_dict.keys() and '128' in enc_feat_dict.keys() and '256' in enc_feat_dict.keys():
            x = self.generator(x, enc_feat_dict['64'], enc_feat_dict['128'], enc_feat_dict['256'])
        return x #, codebook_loss, quant_stats