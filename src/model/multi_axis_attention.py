import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
import math

class MultiAxisAttention(nn.Module):
    def __init__(self, args, channels=64, k_size=3, reduction=4, conv=common.default_conv, res_scale=1, axis=1):
        # axis == 1: grid, axis == 2: block
        super(MultiAxisAttention,self).__init__()
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = common.BasicBlock(channels, channels//reduction, k_size, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(channels, channels, 1, bn=False, act=None)

        if args.multi_axis_atten_method=='hybrid_parallel' and args.multi_axis_atten_fusion=='cat':
            self.conv_fusion = common.BasicBlock(channels*2, channels, 1, bn=False, act=None)
        self.args = args
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.axis = axis
        if self.args.with_layer_norm_channel:
            self.layer_norm_channel = torch.nn.LayerNorm(channels)

    def forward(self, x):
        x_input = x
        if self.args.multi_axis_atten_method=='pure':
            #print('x.shape: ', x.shape)
            N,_,H,W = x.shape

            b_H = math.ceil(math.sqrt(H))
            if self.args.no_perfect_balanced_blocking:
                n_H = math.ceil(H/b_H)
            else:
                n_H = b_H

            b_W = math.ceil(math.sqrt(W))
            if self.args.no_perfect_balanced_blocking:
                n_W = math.ceil(W/b_W)
            else:
                n_W = b_W
            block_size = b_H*b_W

            pad_H = b_H*n_H
            pad_W = b_W*n_W
            pad_L = pad_H*pad_W

            x_embed = self.conv_match(x)
            y_embed = self.conv_assembly(x)            
            L = H*W
            C = x_embed.shape[-3]
            #print('x_embed.shape 1: ', x_embed.shape)

            padding = pad_L - L
            if padding:
                padding_w = pad_W-W
                padding_h = pad_H-H
                last_x_w = W%b_W
                last_x_h = H%b_H
                last_y_w = W%b_W
                last_y_h = H%b_H
                #print('x_embed.shape: ', x_embed.shape)
                #print('padding_w:', padding_w)
                if padding_w:
                    pad_x_w = torch.zeros_like(x_embed[:,:,:,-padding_w:])
                    #print('x_embed.shape: ', x_embed.shape)
                    x_embed = torch.cat([x_embed, pad_x_w], dim=3)
                    #print('x_embed.shape: ', x_embed.shape)
        
                if padding_h:
                    pad_x_h = torch.zeros_like(x_embed[:,:,-padding_h:,:])
                    x_embed = torch.cat([x_embed, pad_x_h], dim=2)
                    #print('x_embed.shape: ', x_embed.shape)

                if padding_w:
                    pad_y_w = torch.zeros_like(y_embed[:,:,:,-padding_w:])
                    y_embed = torch.cat([y_embed, pad_y_w], dim=3)
                if padding_h:
                    pad_y_h = torch.zeros_like(y_embed[:,:,-padding_h:,:])
                    y_embed = torch.cat([y_embed, pad_y_h], dim=2)

            #print('x_embed.shape 1: ', x_embed.shape)

            ##print('C: ', C)
            x_embed = torch.nn.Unfold((b_H, b_W), 1, 0, (b_H, b_W))(x_embed)
            ##print('x_embed.shape 4: ', x_embed.shape)
            x_embed = x_embed.reshape((-1, C, block_size, pad_L//block_size))
            ##print('x_embed.shape 5: ', x_embed.shape
            if self.axis==1:
                x_embed = x_embed.permute(0,2,3,1)
            else:
                x_embed = x_embed.permute(0,3,2,1)

            
            y_embed = torch.nn.Unfold((b_H, b_W), 1, 0, (b_H, b_W))(y_embed)
            y_embed = y_embed.reshape(-1, C*self.reduction, block_size, pad_L//block_size)
            if self.axis==1:
                y_embed = y_embed.permute(0,2,3,1)
            else:
                y_embed = y_embed.permute(0,3,2,1)

            x_match = F.normalize(x_embed, p=2, dim=-1,eps=5e-5)

            #unormalized attention score
            raw_score = torch.einsum('bkie,bkje->bkij', x_embed, x_match) #[N, num_blocks, block_size, block_size*3]

            if self.args.atten_noself:
                SELF_LOC_RAW_SCORE = -1e20
                #if False:
                #if True:
                if not self.args.new_pytorch_version_diagonal_scatter:
                    if self.args.no_perfect_balanced_blocking:
                        if self.axis == 1:
                            diag_indices = torch.eye(pad_L//block_size, pad_L//block_size).to(self.device)
                        else:
                            diag_indices = torch.eye(block_size, block_size).to(self.device)
                    else:
                        diag_indices = torch.eye(block_size, block_size).to(self.device)
                    diag_indices = diag_indices.unsqueeze(0)
                    diag_indices = diag_indices.unsqueeze(0).byte()
                    #print(diag_indices.shape)
                    raw_score = raw_score.masked_fill_(diag_indices, SELF_LOC_RAW_SCORE)
                else:
                    diag_raw_scores = torch.ones(self.args.chunk_size)*SELF_LOC_RAW_SCORE
                    diag_raw_scores = diag_raw_scores.unsqueeze(0)
                    diag_raw_scores = diag_raw_scores.unsqueeze(0)
                    N, B, _, _ = raw_score.shape
                    diag_raw_scores = diag_raw_scores.expand(N, B, -1)
                    raw_score = torch.diagonal_scatter(raw_score, diag_raw_scores, 0, 2, 3)

            #softmax
            bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
            score = torch.exp(raw_score- bucket_score) #(after softmax)
            ###print('score: ', score[0][0])
            ###print('max score: ', torch.max(score[0][0][0]))
            ###print('min score: ', torch.min(score[0][0][0]))
            ###print('argmax score: ', torch.argmax(score[0][0][0]))
            ###print('argmin score: ', torch.argmin(score[0][0][0]))

            #attention
            ##print('score.shape: ', score.shape)
            ##print('y_embed.shape: ', y_embed.shape)
            ret = torch.einsum('bkij,bkje->bkie', score, y_embed) #[N, block_size, num_blocks, C]
            
            if self.axis==1:
                ret = ret.permute((0,3,1,2))
            else:
                ret = ret.permute((0,3,2,1))
            ret = torch.reshape(ret,(N,C*self.reduction*block_size,pad_L//block_size))
            ret = torch.nn.Fold((pad_H, pad_W), (b_H, b_W), 1, 0, (b_H, b_W))(ret)

            ##print('ret.shape: ', ret.shape)

            if padding:
                if padding_h:
                    ret = ret[:,:,:-padding_h,:]
                ##print('ret.shape: ', ret.shape)
                if padding_w:
                    ret = ret[:,:,:,:-padding_w]
                ##print('ret.shape: ', ret.shape)

            if self.args.with_layer_norm_channel:
                ret = ret.permute(0,2,3,1)
                ret = self.layer_norm_channel(ret)
                ret = ret.permute(0,3,1,2)

            if not self.args.no_with_mxba_scale:
                ret = ret*self.res_scale+x_input

            return ret

        elif self.args.multi_axis_atten_method=='hybrid_parallel':
            #print('x.shape: ', x.shape)
            N,_,H,W = x.shape

            b_H = math.ceil(math.sqrt(H))
            n_H = b_H
            b_W = math.ceil(math.sqrt(W))
            n_W = b_W
            block_size = b_H*b_W

            pad_H = b_H*n_H
            pad_W = b_W*n_W
            pad_L = pad_H*pad_W

            x_embed = self.conv_match(x)
            y_embed = self.conv_assembly(x)            
            L = H*W
            C = x_embed.shape[-3]
            #print('x_embed.shape 1: ', x_embed.shape)

            padding = pad_L - L
            if padding:
                padding_w = pad_W-W
                padding_h = pad_H-H
                #print('x_embed.shape: ', x_embed.shape)
                #print('padding_w:', padding_w)
                if padding_w:
                    pad_x_w = torch.zeros_like(x_embed[:,:,:,-padding_w:])
                    #print('x_embed.shape: ', x_embed.shape)
                    x_embed = torch.cat([x_embed, pad_x_w], dim=3)
                    #print('x_embed.shape: ', x_embed.shape)
                if padding_h:
                    pad_x_h = torch.zeros_like(x_embed[:,:,-padding_h:,:])
                    x_embed = torch.cat([x_embed, pad_x_h], dim=2)
                    #print('x_embed.shape: ', x_embed.shape)
                if padding_w:
                    pad_y_w = torch.zeros_like(y_embed[:,:,:,-padding_w:])
                    y_embed = torch.cat([y_embed, pad_y_w], dim=3)
                if padding_h:
                    pad_y_h = torch.zeros_like(y_embed[:,:,-padding_h:,:])
                    y_embed = torch.cat([y_embed, pad_y_h], dim=2)

            #print('x_embed.shape 1: ', x_embed.shape)
            ##print('C: ', C)
            x_embed = torch.nn.Unfold((b_H, b_W), 1, 0, (b_H, b_W))(x_embed)
            ##print('x_embed.shape 4: ', x_embed.shape)
            x_embed = x_embed.reshape((-1, C, block_size, pad_L//block_size))
            ##print('x_embed.shape 5: ', x_embed.shape
            x_embed_a1 = x_embed.permute(0,2,3,1)
            x_embed_a2 = x_embed.permute(0,3,2,1)
            x_embed = torch.cat([x_embed_a1, x_embed_a2], dim=0)

            y_embed = torch.nn.Unfold((b_H, b_W), 1, 0, (b_H, b_W))(y_embed)
            y_embed = y_embed.reshape(-1, C*self.reduction, block_size, pad_L//block_size)
            y_embed_a1 = y_embed.permute(0,2,3,1)
            y_embed_a2 = y_embed.permute(0,3,2,1)
            y_embed = torch.cat([y_embed_a1, y_embed_a2], dim=0)

            x_match_a1 = F.normalize(x_embed_a1, p=2, dim=-1,eps=5e-5)
            x_match_a2 = x_match_a1.permute(0,2,1,3)
            x_match = torch.cat([x_match_a1, x_match_a2], dim=0)

            #unormalized attention score
            raw_score = torch.einsum('bkie,bkje->bkij', x_embed, x_match) #[N, num_blocks, block_size, block_size*3]

            if self.args.atten_noself:
                SELF_LOC_RAW_SCORE = -1e20
                if not self.args.new_pytorch_version_diagonal_scatter:
                    if self.args.no_perfect_balanced_blocking:
                        if self.axis == 1:
                            diag_indices = torch.eye(pad_L//block_size, pad_L//block_size).to(self.device)
                        else:
                            diag_indices = torch.eye(block_size, block_size).to(self.device)
                    else:
                        diag_indices = torch.eye(block_size, block_size).to(self.device)
                    diag_indices = diag_indices.unsqueeze(0)
                    diag_indices = diag_indices.unsqueeze(0).byte()
                    #print(diag_indices.shape)
                    raw_score = raw_score.masked_fill_(diag_indices, SELF_LOC_RAW_SCORE)
                else:
                    diag_raw_scores = torch.ones(self.args.chunk_size)*SELF_LOC_RAW_SCORE
                    diag_raw_scores = diag_raw_scores.unsqueeze(0)
                    diag_raw_scores = diag_raw_scores.unsqueeze(0)
                    N, B, _, _ = raw_score.shape
                    diag_raw_scores = diag_raw_scores.expand(N, B, -1)
                    raw_score = torch.diagonal_scatter(raw_score, diag_raw_scores, 0, 2, 3)

            #softmax
            bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
            score = torch.exp(raw_score- bucket_score) #(after softmax)
            ###print('score: ', score[0][0])
            ###print('max score: ', torch.max(score[0][0][0]))
            ###print('min score: ', torch.min(score[0][0][0]))
            ###print('argmax score: ', torch.argmax(score[0][0][0]))
            ###print('argmin score: ', torch.argmin(score[0][0][0]))

            #attention
            ##print('score.shape: ', score.shape)
            ##print('y_embed.shape: ', y_embed.shape)
            ret = torch.einsum('bkij,bkje->bkie', score, y_embed) #[N, block_size, num_blocks, C]
            ret_a1, ret_a2 = torch.split(ret, [N, N], dim=0)
            ret_a1 = ret_a1.permute((0,3,1,2))
            ret_a2 = ret_a2.permute((0,3,2,1))
            if self.args.multi_axis_atten_fusion=='cat':
                ret = torch.cat([ret_a1, ret_a2], dim=1)
                ret = self.conv_fusion(ret)
            else:
                ret = ret_a1 + ret_a2
            ret = torch.reshape(ret,(N,C*self.reduction*block_size,pad_L//block_size))
            ret = torch.nn.Fold((pad_H, pad_W), (b_H, b_W), 1, 0, (b_H, b_W))(ret)
            
            ##print('ret.shape: ', ret.shape)

            if padding:
                if padding_h:
                    ret = ret[:,:,:-padding_h,:]
                ##print('ret.shape: ', ret.shape)
                if padding_w:
                    ret = ret[:,:,:,:-padding_w]
                ##print('ret.shape: ', ret.shape)

            if self.args.with_layer_norm_channel:
                ret = ret.permute(0,2,3,1)
                ret = self.layer_norm_channel(ret)
                ret = ret.permute(0,3,1,2)

            if not self.args.no_with_mxba_scale:
                ret = ret*self.res_scale+x_input

            return ret
