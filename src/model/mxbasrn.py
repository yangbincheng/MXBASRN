from model import common

import torch.nn as nn
import torch
from model import multi_axis_attention
import math

def make_model(args, parent=False):
    return MXBASRN(args)

no_attention = False
no_rir = False
class AttentionResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        out = self.fn(x)
        return out + x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, groups=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True, groups=groups),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, groups=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, groups=groups))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        global no_attention
        if not no_attention:
            modules_body.append(CALayer(n_feat, reduction, groups=groups))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, groups=1):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, groups=groups) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size, groups=groups))
        self.body = nn.Sequential(*modules_body)
        global no_rir
        self.non_rir = no_rir

    def forward(self, x):
        res = self.body(x)
        if not self.non_rir:
            res += x
        return res

class MXBASRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MXBASRN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.args = args        
        global no_attention
        no_attention = args.no_attention

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args, args.rgb_range, rgb_mean, rgb_std)
        
        # define head module

        modules_head = [conv(args.n_colors*1, n_feats, kernel_size, groups=1)]

        # define body module
        modules_body_feat = nn.ModuleList() 
        m_list = []
        for i in range(args.n_resgroups):
            m_list.append(ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, groups=1))

            if args.use_multi_axis_atten:
                if args.full_attention:
                    if self.args.multi_axis_atten_method=='pure':
                        if self.args.multi_axis_atten_method_only_grid:
                            m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                        else:
                            m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                        if self.args.multi_axis_atten_method_only_block:
                            m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                        else:
                            m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                    else:
                        for _ in range(args.num_multi_axis_atten):
                            m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale))
                elif args.configurable_attention:
                    if i+args.configurable_attention_num>=args.n_resgroups: 
                        if self.args.multi_axis_atten_method=='pure':
                            if self.args.multi_axis_atten_method_only_grid:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                            else:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                            if self.args.multi_axis_atten_method_only_block:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                            else:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                        else:
                            for _ in range(args.num_multi_axis_atten):
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale))
                elif args.fix_position_attention:
                    fix_positions = [ int(_) for _ in args.fix_position_attention_list.split('+')]
                    if i in fix_positions: 
                        if self.args.multi_axis_atten_method=='pure':
                            if self.args.multi_axis_atten_method_only_grid:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                            else:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                            if self.args.multi_axis_atten_method_only_block:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=2))
                            else:
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale, axis=1))
                        else:
                            for _ in range(args.num_multi_axis_atten):
                                m_list.append(multi_axis_attention.MultiAxisAttention( args, channels=n_feats, reduction=4, res_scale=args.res_scale))
                else:
                    pass

        modules_body_feat.append(nn.Sequential(*m_list))

        modules_body_conv = [conv(n_feats, n_feats, kernel_size)]

        # define tail module
        n_outfeats_us = n_feats
        if args.small_upsampling_head:
            if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
                for _ in range(int(math.log(scale, 2))):
                    n_outfeats_us = int(math.ceil(n_outfeats_us/4))
                    if args.small_upsampling_head_with_width_threshold:
                        if n_outfeats_us < args.small_upsampling_head_with_width_threshold_value:
                            n_outfeats_us = args.small_upsampling_head_with_width_threshold_value
            elif scale == 3:
                n_outfeats_us = int(math.ceil(n_outfeats_us/9))
                if args.small_upsampling_head_with_width_threshold:
                    if n_outfeats_us < args.small_upsampling_head_with_width_threshold_value:
                        n_outfeats_us = args.small_upsampling_head_with_width_threshold_value
            else:
                raise NotImplementedError

        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False, small_upsampling_head=args.small_upsampling_head, small_upsampling_head_with_width_threshold=args.small_upsampling_head_with_width_threshold, small_upsampling_head_with_width_threshold_value=args.small_upsampling_head_with_width_threshold_value),
            conv(n_outfeats_us, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args, args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body_feat = modules_body_feat
        self.body_conv = nn.Sequential(*modules_body_conv)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        if self.args.shift_mean:
            x = self.sub_mean(x)

        x = self.head(x)
        x_head = x

        x = self.body_feat[0](x)
        res = self.body_conv(x)
        res += x_head
        x = self.tail(res)

        if self.args.shift_mean:
            x = self.add_mean(x)

        return x
        
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
