import math
import numpy as np
import torch
from torch import nn, Tensor, einsum
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.utils import get_root_logger, GradLayer, tensor2img,imwrite
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from einops.layers.torch import Rearrange

from basicsr.utils.motion_estimator_util import make_coordinate_grid
import random


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1, mask=None):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        if mask is not None:
            scale = mask * scale
            shift = mask * shift
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out

class Fuse_feat_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

    def forward(self, enc_feat, dec_feat):
        res = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        out = dec_feat + res
        return out
 

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_conv=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP -> Conv2D
        self.conv1 = nn.Conv2d(embed_dim, dim_conv, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(dim_conv, embed_dim, kernel_size=3, padding=1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, context, shape, use_spatial_attn=False, use_residual=False,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        b, _, h, w = shape
        if use_residual:
            input = tgt
        # self attention
        tgt2 = self.norm1(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(tgt2, query_pos)
        v = tgt2 
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        if use_spatial_attn:
            tgt = tgt[:h*w,...] + self.dropout1(tgt2[:h*w,...])
        else:
            tgt = tgt + self.dropout1(tgt2)

        # cross attention
        tgt2 = self.norm2(tgt)
        if use_spatial_attn:
            q = self.with_pos_embed(tgt2, query_pos[:h*w,...])
        else:
            q = self.with_pos_embed(tgt2, query_pos)
        k = v = context
        tgt2 = self.cross_attn(q, k, value=v)[0]
        tgt = tgt + self.dropout2(tgt2)

        # ffn-> conv
        tgt2 = self.norm3(tgt)
        tgt2 = self.conv2(self.dropout(self.activation(self.conv1(tgt2.permute(1,2,0).reshape(b,self.embed_dim,h,w)))))
        tgt2 = tgt2.reshape(b,self.embed_dim, h*w).permute(2,0,1)

        tgt = tgt + self.dropout3(tgt2)
        if use_residual:
            tgt = tgt + input
        return tgt

# from MRFA
class BasicMotionEncoder(nn.Module):
    def __init__(self, motion_dim):
        super(BasicMotionEncoder, self).__init__()
        #cor_planes = num_levels * (2*radius + 1)**2
        self.convc1 = nn.Conv2d(motion_dim, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+96, 128-2, 3, padding=1)
        
    def forward(self, delta_flow, motion_feat):
        cor = F.relu(self.convc1(motion_feat))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(delta_flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, delta_flow], dim=1)

# from MRFA
class RefineFlow(nn.Module):
    def __init__(self):
        super(RefineFlow, self).__init__()
        self.convc1 = nn.Conv2d(192, 128, 3, padding=1)
        # self.convc2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 2, 3, padding=1)
        self.convo1 = nn.Conv2d(256, 128, 3, padding=1)
        self.convo2 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, m_f, warp_f):
        c = F.relu(self.convc1(warp_f))
        ## c = F.relu(self.convc2(c))
        inp = torch.cat([m_f,c],dim=1)
        flow = self.conv2(F.relu(self.conv1(inp)))
        occ = self.convo2(F.relu(self.convo1(inp)))
        out = torch.cat([flow,occ],dim=1)
        return out, inp
        # return flow, inp

@ARCH_REGISTRY.register()
class AppMotionCompFormer(AutoEncoder):
    def __init__(self, img_size=256, nf=64, ch_mult=[1, 2, 2, 4], res_blocks=2, attn_resolutions=[32],
                quantizer_type="nearest", beta=0.25,
                codebook_size_motion=1024, embed_dim_motion=32, codebook_size_app=1024, embed_dim_app=256,
                n_head=8, dim_embd_motion=32, n_layers_motion=2, dim_embd_app=256, n_layers_app=2, split=1,
                num_kp=15,
                with_position_emb=True,
                warp_s_d_kp_query=True, # use warped source and driving keypoints as query
                MRFA_motion_enc=True,
                motion_codebook_split=True,
                detach_motion_query=True,
                multiscale_feature_fusion=True,
                multiscale_sft = True,
                app_codebook_split=True,
                wo_motion_cdbk_share=False, # equally splitting motion codebook
                wo_app_cdbk_share=False, # equally splitting appearance codebook
                connect_list=['64', '128', '256'], 
                connect_app_list=['32', '64', '128', '256'],
                fix_modules=[], ae_path=None):
        super(AppMotionCompFormer, self).__init__(img_size=img_size, nf=nf, ch_mult=ch_mult, res_blocks=res_blocks, attn_resolutions=attn_resolutions)

        self.with_position_emb = with_position_emb

        self.warp_s_d_kp_query = warp_s_d_kp_query
        self.MRFA_motion_enc = MRFA_motion_enc
        self.motion_codebook_split = motion_codebook_split
        
        self.detach_motion_query = detach_motion_query
        
        self.multiscale_feature_fusion = multiscale_feature_fusion
        self.multiscale_sft = multiscale_sft

        self.app_codebook_split = app_codebook_split and len(connect_app_list)>1
        
        self.wo_motion_cdbk_share = wo_motion_cdbk_share
        self.wo_app_cdbk_share = wo_app_cdbk_share

        self.connect_list = connect_list
        self.connect_app_list = connect_app_list

        self.channels = {
            '32': 256,
            '64': 128,
            '128': 128,
            '256': 64,
        }

        if '32' in self.connect_app_list:
            self.app_feat_emb_32 = nn.Conv2d(self.channels['32']//split, dim_embd_app, kernel_size=1, stride=1, padding=0)
            self.to_app_feat_32 = nn.Conv2d(dim_embd_app, self.channels['32']//split, kernel_size=1, stride=1, padding=0)
        if '64' in self.connect_app_list:
            app_patch_embedding_64 = [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2, p2 = 2), nn.Linear(self.channels['64']//split* 2 * 2,dim_embd_app), Rearrange('b n d -> b d n')]
            to_app_feat_64 = [nn.Linear(dim_embd_app, self.channels['64']//split* 2 * 2), Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=64//2, p1 = 2, p2 = 2)]

            self.app_feat_emb_64 = nn.Sequential(*app_patch_embedding_64)
            self.to_app_feat_64 = nn.Sequential(*to_app_feat_64)

        if '128' in self.connect_app_list:
            app_patch_embedding_128 = [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4), nn.Linear(self.channels['128']//split* 4 * 4,dim_embd_app), Rearrange('b n d -> b d n')]
            to_app_feat_128 = [nn.Linear(dim_embd_app, self.channels['128']//split* 4 * 4), Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=128//4, p1 = 4, p2 = 4)]

            self.app_feat_emb_128 = nn.Sequential(*app_patch_embedding_128)
            self.to_app_feat_128 = nn.Sequential(*to_app_feat_128)

        if '256' in self.connect_app_list:
            app_patch_embedding_256 = [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 8, p2 = 8), nn.Linear(self.channels['256']//split* 8 * 8,dim_embd_app), Rearrange('b n d -> b d n')]
            to_app_feat_256 = [nn.Linear(dim_embd_app, self.channels['256']//split* 8 * 8), Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=256//8, p1 = 8, p2 = 8)]
            
            self.app_feat_emb_256 = nn.Sequential(*app_patch_embedding_256)
            self.to_app_feat_256 = nn.Sequential(*to_app_feat_256)
        
        if quantizer_type == "nearest":
            self.beta = beta 
            self.codebook_size_app = codebook_size_app
            self.embed_dim_app = embed_dim_app

            self.quantize_app = VectorQuantizer(self.codebook_size_app, self.embed_dim_app, self.beta)

        # fuse_convs_dict
        if self.multiscale_sft:
            self.fuse_convs_dict = nn.ModuleDict()
        if self.multiscale_feature_fusion:
            self.fuse_ms_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            if self.multiscale_sft:
                self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch) # CFT
            if self.multiscale_feature_fusion:
                self.fuse_ms_dict[f_size] = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        
        if ae_path is not None:
            self.load_state_dict(
                torch.load(ae_path, map_location='cpu')['params_ema'])
        
        if with_position_emb:
            self.position_emb_app = nn.Parameter(torch.zeros(32*32, dim_embd_app))
            self.position_emb_motion = nn.Parameter(torch.zeros(32*32, dim_embd_motion))
            
        else:
            self.position_emb_app = None
            self.position_emb_motion = None
        
        if quantizer_type == "nearest":
            self.codebook_size_motion = codebook_size_motion
            self.embed_dim_motion = embed_dim_motion

            self.quantize_motion = VectorQuantizer(self.codebook_size_motion, self.embed_dim_motion, self.beta)
        
        # motion
        self.n_layers_motion = n_layers_motion
        self.dim_embd_motion = dim_embd_motion
        self.dim_conv_motion = dim_embd_motion*2

        motion_emb_layers = [nn.Conv2d(2, dim_embd_motion, kernel_size=3, stride=1, padding=1), Downsample(dim_embd_motion), ResBlock(dim_embd_motion,dim_embd_motion)]

        self.motion_emb = nn.Sequential(*motion_emb_layers) 
        self.motion_block = nn.Sequential(*[TransformerLayer(embed_dim=dim_embd_motion, nhead=n_head, dim_conv=self.dim_conv_motion, dropout=0.0) 
                                    for _ in range(self.n_layers_motion)])
        
        to_motion_layers = [Upsample(dim_embd_motion), ResBlock(dim_embd_motion,dim_embd_motion), normalize(dim_embd_motion), nn.Conv2d(dim_embd_motion, 2, kernel_size=3, stride=1, padding=1)]

        self.to_motion = nn.Sequential(*to_motion_layers) 
        
        if self.MRFA_motion_enc:
            self.BasicMotionEncoder = BasicMotionEncoder(dim_embd_motion)
            self.to_context = nn.ModuleList()

            for i in ['32', '64', '128']:
                self.to_context.append(nn.Conv2d(self.channels[i], 192, 1, padding=0))
            if '256' in self.connect_list:
                self.to_context.append(nn.Conv2d(self.channels['256'], 192, 1, padding=0))

            self.refine = RefineFlow()
        
        # appearance
        self.split = split
        self.n_layers_app = n_layers_app
        self.dim_embd_app = dim_embd_app
        self.dim_conv_app = dim_embd_app*2

        self.app_block = nn.Sequential(*[TransformerLayer(embed_dim=dim_embd_app, nhead=n_head, dim_conv=self.dim_conv_app, dropout=0.0) 
                                    for _ in range(self.n_layers_app)])


        if self.warp_s_d_kp_query:
            self.warped_source_enc_32 = nn.Conv2d(self.channels['32'], dim_embd_motion, kernel_size=1, stride=1, padding=0)
            self.warped_source_enc_64 = nn.Conv2d(self.channels['64'], dim_embd_motion, kernel_size=1, stride=1, padding=0)
            self.warped_source_enc_128 = nn.Conv2d(self.channels['128'], dim_embd_motion, kernel_size=1, stride=1, padding=0)
            if '256' in self.connect_list:
                self.warped_source_enc_256 = nn.Conv2d(self.channels['256'], dim_embd_motion, kernel_size=1, stride=1, padding=0)

            self.driving_kp_enc = nn.Conv2d(num_kp, dim_embd_motion, kernel_size=1, stride=1, padding=0)

            self.motion_query_enc_1 = nn.Conv2d(dim_embd_motion*2, dim_embd_motion, kernel_size=1, stride=1, padding=0)
            self.motion_query_enc_2 = nn.Conv2d(dim_embd_motion*2, dim_embd_motion, kernel_size=1, stride=1, padding=0)

        self.fuse_encoder_block = {'256':2, '128':5, '64':8, '32':11}

        self.fuse_generator_block = {'32':6, '64': 9, '128':12, '256':15}

        if fix_modules is not None:
            for module in fix_modules:
                if module in ['position_emb_app', 'position_emb_motion']:
                    param = getattr(self, module)
                    param.requires_grad = False
                    continue
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out
       
    def encode_driving(self, x):
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list+['32']]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        return enc_feat_dict
    
    def motion_codebook_compensation(self, motion, query_feat=None, warp_source_feat=None, scale=None, inference=False):
        dict = {32: 1, 64:2, 128:3, 256:4}
        b,h,w,c = motion.shape
        m = motion.permute(0,3,1,2)
        m_feat = self.motion_emb(m.detach())

        if not inference:
            if scale is not None:
                if self.wo_motion_cdbk_share:
                    quant_motion, codebook_loss_motion, quant_stats_motion = self.quantize_motion(m_feat, dict[scale]/(1.0+len(self.connect_list)), (dict[scale]-1)/(1.0+len(self.connect_list)))
                else:
                    quant_motion, codebook_loss_motion, quant_stats_motion = self.quantize_motion(m_feat, dict[scale]/(1.0+len(self.connect_list)))
            else:
                quant_motion, codebook_loss_motion, quant_stats_motion = self.quantize_motion(m_feat)
        
        if self.warp_s_d_kp_query:
            if query_feat.shape[2]!=m_feat.shape[2] or query_feat.shape[3]!=m_feat.shape[3]:
                query_feat = F.interpolate(query_feat, size=(m_feat.shape[2], m_feat.shape[3]), mode='bilinear', align_corners=True)
            query_emb = self.motion_query_enc_2(torch.cat((m_feat, query_feat), dim=1))
            query_emb = query_emb.reshape(b,self.dim_embd_motion, 32*32).permute(2,0,1)
        else:
            m_feat = m_feat.reshape(b,self.dim_embd_motion, 32*32).permute(2,0,1)
            query_emb = m_feat
        if self.with_position_emb:
            pos_emb = self.position_emb_motion.unsqueeze(1).repeat(1,b,1)
        else:
            pos_emb = None
        for block in self.motion_block:
            if scale is not None:
                if self.wo_motion_cdbk_share:
                    query_emb = block(query_emb, self.quantize_motion.embedding.weight[self.codebook_size_motion//int((1.0+len(self.connect_list)))*(dict[scale]-1):self.codebook_size_motion//int((1.0+len(self.connect_list)))*dict[scale], ...].reshape(self.codebook_size_motion//int((1.0+len(self.connect_list))), 1, self.embed_dim_motion).repeat(1,b,1),shape=(b,c,32,32),query_pos=pos_emb)
                else:
                    query_emb = block(query_emb, self.quantize_motion.embedding.weight[:self.codebook_size_motion//int((1.0+len(self.connect_list)))*dict[scale], ...].reshape(self.codebook_size_motion//int((1.0+len(self.connect_list)))*dict[scale], 1, self.embed_dim_motion).repeat(1,b,1),shape=(b,c,32,32),query_pos=pos_emb)
            else:
                query_emb = block(query_emb, self.quantize_motion.embedding.weight.reshape(self.codebook_size_motion, 1, self.embed_dim_motion).repeat(1,b,1),shape=(b,c,32,32),query_pos=pos_emb)

        query_emb = query_emb.permute(1,2,0).reshape(b,self.dim_embd_motion,32,32)

        if self.MRFA_motion_enc:
            motion_f = query_emb 
            if h!=motion_f.shape[2] or w!=motion_f.shape[3]:
                motion_f = F.interpolate(motion_f, size=(h, w), mode='bilinear', align_corners=True)
            m_f = self.BasicMotionEncoder(motion.permute(0,3,1,2), motion_f)
            warp_f = F.relu(self.to_context[int(math.log(warp_source_feat.shape[-1], 2))-5](warp_source_feat))
            if h!=warp_f.shape[2] or w!=warp_f.shape[3]:
                warp_f = F.interpolate(warp_f, size=(h, w), mode='bilinear', align_corners=True)
            m_res, _= self.refine(m_f,warp_f)
        else:
            m_res = self.to_motion(query_emb)

        if inference:
            return [m_res.permute(0,2,3,1)]
        else:
            m_recon = self.to_motion(quant_motion).permute(0,2,3,1)
            return [m_res.permute(0,2,3,1), m_recon, codebook_loss_motion]

    def app_codebook_loss(self, x, weight=1, fuse_list=None): # x = driving
        dict = {32: 1, 64:2, 128:3, 256:4} 
        split_num = float(len(self.connect_app_list))

        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_app_list]

        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        
        app_recon_list = []
        codebook_loss_app_list = []
        quant_app_recon_img_list = []
        for w in self.connect_app_list:
            if self.split == 2:
                feat_com = enc_feat_dict[w][:,1::2,...]
            else:
                feat_com = enc_feat_dict[w]
            app_feat = eval('self.app_feat_emb_'+w)(feat_com)
            if eval(w) > 32:
                app_feat = app_feat.reshape(app_feat.shape[0],app_feat.shape[1], 32, 32)
            if self.app_codebook_split:
                if self.wo_app_cdbk_share:
                    quant_app, codebook_loss_app, quant_stats_app = self.quantize_app(app_feat, dict[feat_com.shape[-1]]/split_num, (dict[feat_com.shape[-1]]-1)/split_num)
                else:
                    quant_app, codebook_loss_app, quant_stats_app = self.quantize_app(app_feat, dict[feat_com.shape[-1]]/split_num)
            else:
                quant_app, codebook_loss_app, quant_stats_app = self.quantize_app(app_feat)
            if eval(w) > 32:
                app_recon = eval('self.to_app_feat_'+w)(quant_app.reshape(quant_app.shape[0], quant_app.shape[1], 32*32).permute(0,2,1))
                app_feat_original = eval('self.to_app_feat_'+w)(app_feat.reshape(quant_app.shape[0], quant_app.shape[1], 32*32).permute(0,2,1))
            else:
                app_recon = eval('self.to_app_feat_'+w)(quant_app)
                app_feat_original = eval('self.to_app_feat_'+w)(app_feat)
            
            app_recon_list.append([app_recon, app_feat_original, quant_app, app_feat, feat_com])
            codebook_loss_app_list.append(codebook_loss_app)

        return app_recon_list, codebook_loss_app_list

    
    def app_codebook_compensation(self, feat, motion=None, source_feat=None, visualize_app_feat=False, vis_original_size=True):
        dict = {32: 1, 64:2, 128:3, 256:4}
        
        split_num = float(len(self.connect_app_list))

        if self.split == 2:
            feat_same, feat_com = feat[:,::2,...],feat[:,1::2,...]
        else:
            feat_com = feat
        b,c,h_f,w_f = feat_com.shape

        h = 32
        w = 32

        flag = 0
        motion = motion.permute(0,3,1,2)
        motion = F.interpolate(motion, size=(h, w), mode='bilinear', align_corners=True)
        motion = motion.reshape(b,2,h*w)

        motion_ignore = (motion>1)+(motion<(-1))
        motion_ignore = motion_ignore[:,0,:] + motion_ignore[:,1,:]

        app_feat = eval('self.app_feat_emb_{}'.format(w_f))(feat_com).reshape(b,self.dim_embd_app, h*w).permute(2,0,1) # (b,c,h*w) -> (h*w,b,c)

        if visualize_app_feat:
            if vis_original_size:
                query_feat = feat_com
            else:
                query_feat = app_feat.permute(1,2,0).reshape(b,self.dim_embd_app,h,w)
        query_emb = app_feat

        if self.with_position_emb:
            pos_emb = self.position_emb_app.unsqueeze(1).repeat(1,b,1)
        else:
            pos_emb = None

        for block in self.app_block:
            if flag == 0:
                if self.app_codebook_split:
                    if self.wo_app_cdbk_share:
                        query_emb = block(query_emb, self.quantize_app.embedding.weight[self.codebook_size_app//int(split_num)*(dict[w_f]-1):self.codebook_size_app//int(split_num)*dict[w_f], ...].reshape(self.codebook_size_app//int(split_num), 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w), tgt_key_padding_mask=motion_ignore, query_pos=pos_emb)
                    else:
                        query_emb = block(query_emb, self.quantize_app.embedding.weight[:self.codebook_size_app//int(split_num)*dict[w_f], ...].reshape(self.codebook_size_app//int(split_num)*dict[w_f], 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w), tgt_key_padding_mask=motion_ignore, query_pos=pos_emb)
                else:
                    query_emb = block(query_emb, self.quantize_app.embedding.weight.reshape(self.codebook_size_app, 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w), tgt_key_padding_mask=motion_ignore, query_pos=pos_emb)
                flag = flag+1
            else:
                if self.app_codebook_split:
                    if self.wo_app_cdbk_share:
                        query_emb = block(query_emb, self.quantize_app.embedding.weight[self.codebook_size_app//int(split_num)*(dict[w_f]-1):self.codebook_size_app//int(split_num)*dict[w_f], ...].reshape(self.codebook_size_app//int(split_num), 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w),query_pos=pos_emb)
                    else:
                        query_emb = block(query_emb, self.quantize_app.embedding.weight[:self.codebook_size_app//int(split_num)*dict[w_f], ...].reshape(self.codebook_size_app//int(split_num)*dict[w_f], 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w),query_pos=pos_emb)
                else:
                    query_emb = block(query_emb, self.quantize_app.embedding.weight.reshape(self.codebook_size_app, 1, self.embed_dim_app).repeat(1,b,1),shape=(b,c,h,w),query_pos=pos_emb)
        
        
        if w_f == 32:
            app_feat = eval('self.to_app_feat_{}'.format(w_f))(query_emb.permute(1,2,0).reshape(b,self.dim_embd_app,h,w))
        else:
            app_feat = eval('self.to_app_feat_{}'.format(w_f))(query_emb.permute(1,0,2))
        
        if visualize_app_feat:
            if vis_original_size:
                comp_feat = app_feat
            else:
                comp_feat = query_emb.permute(1,2,0).reshape(b,self.dim_embd_app,h,w)

        if self.split ==2:
            out = torch.stack((feat_same, app_feat), dim=2)
            app_feat = out.view(b,c*2,h,w)
        if visualize_app_feat:
            return app_feat, query_feat, comp_feat
        return app_feat 
    
    def forward(self, x, dense_motion, w=1, 
                inference=False, vis_app_before_comp=False, gt=None, visualize_app_feat=False):

        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        motion_list = [dense_motion['deformation']]

        out_occ = []

        # motion compensation 
        # make grid
        xx = torch.linspace(-1., 1., motion_list[-1].shape[1])
        yy = torch.linspace(-1., 1., motion_list[-1].shape[2])
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing='xy')
        grid = torch.cat((grid_x.unsqueeze(0).unsqueeze(-1), grid_y.unsqueeze(0).unsqueeze(-1)), dim=-1).to(motion_list[-1].device)

        motion_q = motion_list[-1]

        if self.warp_s_d_kp_query:
            warp_source_feat_0 = self.deform_input(x, motion_list[-1])
            driving_kp_feat = F.relu(self.driving_kp_enc(F.interpolate(dense_motion['driving_kp_heatmap'], size=(32, 32), mode='bilinear', align_corners=True)))

            warp_source_feat = F.relu(eval('self.warped_source_enc_'+str(x.shape[-1]))(warp_source_feat_0))
            motion_q_feat = self.motion_query_enc_1(torch.cat((warp_source_feat, driving_kp_feat), dim=1))

            if self.motion_codebook_split:
                m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), motion_q_feat, warp_source_feat_0, x.shape[-1], inference=inference)
            else:
                m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), motion_q_feat, warp_source_feat_0, inference=inference) 
        else:
            m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), inference=inference) 
        
        if inference:
            motion_res = m_list[0]
        else:
            motion_res, m_recon, codebook_loss_motion = m_list
            motion_recon_list = [m_recon/((motion_list[-1].shape[1]-1.)/2.)]
            codebook_loss_motion_list = [codebook_loss_motion]
        
        if self.MRFA_motion_enc:
            d_occ = motion_res[:,:,:,2:].permute(0,3,1,2)
            motion_res = motion_res[:,:,:,0:2]
        res_motion_list = [motion_res/((motion_list[-1].shape[1]-1.)/2.)]
        m_com = motion_list[-1]+res_motion_list[-1]
        motion_list.append(m_com)
        

        if not isinstance(dense_motion['occlusion_map'],list) and self.MRFA_motion_enc:
            occlusion = dense_motion['occlusion_map'] + d_occ

            out_occ.append(F.sigmoid(occlusion))

        # feature warping
        lq_feat = self.deform_input(x, m_com)
        
        if isinstance(dense_motion['occlusion_map'],list):
            lq_feat = self.occlude_input(lq_feat, dense_motion['occlusion_map'][0])
            # enc_feat
            deform_feat_list = [self.occlude_input(self.deform_input(x.detach(), m_com), dense_motion['occlusion_map'][0].detach())]
            occlusion_idx = 1
        else:
            if self.MRFA_motion_enc:
                lq_feat = self.occlude_input(lq_feat, out_occ[0])
                # enc_feat
                deform_feat_list = [self.occlude_input(self.deform_input(x.detach(), m_com), out_occ[0].detach())]
            else:
                lq_feat = self.occlude_input(lq_feat, dense_motion['occlusion_map'])
                # enc_feat
                deform_feat_list = [self.occlude_input(self.deform_input(x.detach(), m_com), dense_motion['occlusion_map'].detach())]

        app_before_comp_list = [lq_feat]

        if visualize_app_feat:
            app_query_feat_list = []
            app_comp_feat_list = []
        
        # appearance compensation 
        if visualize_app_feat:
            lq_feat, query_feat, comp_feat =  self.app_codebook_compensation(lq_feat, motion=m_com, visualize_app_feat=visualize_app_feat)
            app_query_feat_list.append(query_feat)
            app_comp_feat_list.append(comp_feat)
        else:
            lq_feat =  self.app_codebook_compensation(lq_feat, motion=m_com)
        app_comp_list = [lq_feat]

        # ################## Generator ####################
        x = lq_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        if gt is not None and not inference:
            app_recon_list, codebook_loss_app_list = self.app_codebook_loss(gt)
            quant_app_recon_img_list = []
        else:
            app_recon_list = []
            codebook_loss_app_list = []
            quant_app_recon_img_list = []

        if inference:
            x_lr_32 = None
        else:
            x_lr_32 = lq_feat.contiguous()

        if vis_app_before_comp:
            x_before_app_32 = app_before_comp_list[0].contiguous()

        for i, block in enumerate(self.generator.blocks):
            x = block(x) 
            if x_lr_32 is not None and not inference:
                x_lr_32 = block(x_lr_32)
            if vis_app_before_comp:
                x_before_app_32 = block(x_before_app_32)
            if i in fuse_list: # fuse after i-th block
                f_size = str(x.shape[-1])
                if w>0:
                    # motion compensation
                    motion_q = motion_list[-1]
                        
                    if self.warp_s_d_kp_query:
                        warp_source_feat_0 = self.deform_input(enc_feat_dict[f_size], motion_list[-1])

                        warp_source_feat = F.relu(eval('self.warped_source_enc_'+str(x.shape[-1]))(F.interpolate(warp_source_feat_0, size=(32, 32), mode='bilinear', align_corners=True)))
                        motion_q_feat = self.motion_query_enc_1(torch.cat((warp_source_feat, driving_kp_feat), dim=1))

                        if self.motion_codebook_split:
                            m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), motion_q_feat, warp_source_feat_0, x.shape[-1], inference=inference)
                        else:
                            m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), motion_q_feat, warp_source_feat_0, inference=inference) 

                    else:
                        m_list = self.motion_codebook_compensation((motion_q-grid)*((motion_list[-1].shape[1]-1.)/2.), inference=inference)

                    if inference:
                        motion_res = m_list[0]
                    else:
                        motion_res, m_recon, codebook_loss_motion = m_list
                        motion_recon_list.append(m_recon/((motion_list[-1].shape[1]-1.)/2.))
                        codebook_loss_motion_list.append(codebook_loss_motion)
                        
                    if self.MRFA_motion_enc:
                        d_occ = motion_res[:,:,:,2:].permute(0,3,1,2)
                        motion_res = motion_res[:,:,:,0:2]

                    res_motion_list.append(motion_res/((motion_list[-1].shape[1]-1.)/2.))

                    m_com = motion_list[-1]+res_motion_list[-1]
                    motion_list.append(m_com)
                     

                    # feature warping
                    enc_feat_warped = self.deform_input(enc_feat_dict[f_size], m_com)
                    
                    if isinstance(dense_motion['occlusion_map'],list):
                        enc_feat_warped = self.occlude_input(enc_feat_warped, dense_motion['occlusion_map'][occlusion_idx])

                        deform_feat_list.append(self.occlude_input(self.deform_input(enc_feat_dict[f_size].detach(), m_com), dense_motion['occlusion_map'][occlusion_idx].detach()))
                        occlusion_idx += 1
                    else:
                        if self.MRFA_motion_enc:
                            occlusion = out_occ[-1] + d_occ
                            out_occ.append(F.sigmoid(occlusion))
                            enc_feat_warped = self.occlude_input(enc_feat_warped, out_occ[-1])

                            # enc_feat
                            deform_feat_list.append(self.occlude_input(self.deform_input(enc_feat_dict[f_size].detach(), m_com), out_occ[-1].detach()))

                        else:
                            enc_feat_warped = self.occlude_input(enc_feat_warped, dense_motion['occlusion_map'])
                            # enc_feat
                            deform_feat_list.append(self.occlude_input(self.deform_input(enc_feat_dict[f_size].detach(), m_com), dense_motion['occlusion_map'].detach()))


                    # appearance compensation
                    if x.shape[-1]<eval(self.connect_app_list[-1]) + 1:
                        app_before_comp_list.append(enc_feat_warped)
                        
                        if visualize_app_feat:
                            enc_feat_warped, query_feat, comp_feat =  self.app_codebook_compensation(enc_feat_warped, motion=m_com, visualize_app_feat=visualize_app_feat)
                            app_query_feat_list.append(query_feat)
                            app_comp_feat_list.append(comp_feat)
                        else:
                            enc_feat_warped =  self.app_codebook_compensation(enc_feat_warped, motion=m_com)
                        
                        app_comp_list.append(enc_feat_warped)

                    if self.multiscale_sft:
                        x = self.fuse_convs_dict[f_size](enc_feat_warped, x, w)
                    if self.multiscale_feature_fusion:
                        x = x + self.fuse_ms_dict[f_size](enc_feat_warped) # addition
        
        out = x
        out_dict = {}
        out_dict['out'] = out
        out_dict['lq_feat'] = lq_feat
        out_dict['out_occ'] = out_occ # list
        out_dict['deformation_list'] = motion_list
        out_dict['res_deform_list'] = res_motion_list
        if not inference:
            out_dict['out_lr'] = [x_lr_32] 
            out_dict['motion_recon_list'] = motion_recon_list
            out_dict['codebook_loss_motion_list'] = codebook_loss_motion_list
        out_dict['deform_feat_list'] = deform_feat_list
        out_dict['app_comp_list'] = app_comp_list # occlusion_list
        out_dict['app_before_comp_list'] = app_before_comp_list

        if gt is not None and not inference:
            out_dict['app_recon_list'] = app_recon_list
            out_dict['codebook_loss_app_list'] = codebook_loss_app_list

        if visualize_app_feat:
            out_dict['app_query_feat_list'] = app_query_feat_list
            out_dict['app_comp_feat_list'] = app_comp_feat_list
        if vis_app_before_comp:
            out_dict['x_before_app_32'] =  x_before_app_32
        return out_dict