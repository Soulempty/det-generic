#coding:utf-8
import torch.nn as nn
import torch
from time import time
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS

#适应性空间特征融合（Adaptive Spatial Fusion）
class ASF(nn.Module):
    def __init__(self,in_nums=4,channels=256,act='sigmoid'):
        super(ASF,self).__init__()
        self.in_nums = in_nums
        self.channels = channels
        
        self.conv1 = nn.Conv2d(in_nums*channels,channels,1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(channels,in_nums,3,padding=1)
        self.act = nn.Sigmoid() if act == 'sigmoid' else nn.Softmax(dim=1)
    def forward(self,feats):
        feat = torch.cat(feats,dim=1)
        x = self.conv1(feat)
        x = self.relu(x)
        x = self.conv3(x)
        act = self.act(x)
        acts = torch.split(act,1,dim=1)
        fuse_feat = 0
        for i in range(self.in_nums):
            fuse_feat += acts[i]*feats[i]

        return fuse_feat

#适应性通道特征融合（Adaptive Channel Fusion）
class ACF(nn.Module):
    def __init__(self,in_nums=4,channels=256,pool_type='max'):
        super(ACF,self).__init__()
        self.in_nums = in_nums
        self.channels = channels
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.pool_type = pool_type
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_nums*channels,channels//4,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels//4,in_nums*channels,1)
    def forward(self,feats):
        feat = torch.cat(feats,dim=1)
        if self.pool_type == 'avg':
            x = self.gap(feat)
        elif self.pool_type == 'max':
            x = self.gmp(feat)
        else:
            x = self.gap(feat) + self.gmp(feat)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        act = self.sigmoid(x)
        acts = torch.split(act,self.channels,dim=1)
        fuse_feat = 0
        for i in range(self.in_nums):
            fuse_feat += acts[i]*feats[i]

        return fuse_feat

#卷积注意力模块（Convolutional Block Attention Module）
class CBAM(nn.Module):
    def __init__(self,in_nums=4,kernal=7):
        super(CBAM,self).__init__()
        self.in_nums = in_nums
 
        self.acf = ACF(in_nums,pool_type='am')
        self.sigmoid = nn.Sigmoid()

        self.conv_s = nn.Conv2d(2,1,kernal,stride=1,padding=(kernal-1)//2)
        
    def forward(self,feats):
        
        weighted_feat = self.acf(feats)
        avg_c = torch.mean(weighted_feat,dim=1,keepdim=True)
        max_c,_ = torch.max(weighted_feat,dim=1,keepdim=True)
        x = torch.cat([avg_c,max_c],dim=1)
        space_weight = self.sigmoid(self.conv_s(x))
        res = space_weight * weighted_feat
        return  res 

        


@NECKS.register_module()
class MyPAFPN(nn.Module):

    def __init__(self,
                 in_channels,# [64, 128, 256]
                 out_channels,# 256
                 num_outs, #1
                 start_level=0,
                 end_level=-1,
                 use_level=1,
                 ops='cbam',
                 use_down = False,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(MyPAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels) #3
        self.num_outs = num_outs #1
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins #3
 
        self.start_level = start_level
        self.end_level = end_level
        self.use_level = use_level
        self.ops = ops
        self.use_down = use_down
       
        self.lateral_convs = nn.ModuleList()
        if ops == 'add':
            self.fpn_conv = ConvModule(out_channels,out_channels,3,padding=1,act_cfg=act_cfg,inplace=False)
        elif ops == 'asf':
            self.fpn_conv = ASF(in_nums=num_outs)
        elif ops == 'acf':
            self.fpn_conv = ACF(in_nums=num_outs)
        elif ops == 'cbam':
            self.fpn_conv = CBAM(in_nums=num_outs,kernal=7)
        else:
            self.fpn_conv = ConvModule(out_channels*self.num_outs,out_channels,3,padding=1,act_cfg=act_cfg,inplace=False)

        for i in range(self.start_level, self.backbone_end_level):# 1,3
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
        self.last_conv = None
        if self.use_down:
            self.last_conv = ConvModule(out_channels,out_channels,1,act_cfg=act_cfg,inplace=False)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)              
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        
        # step 1: gather multi-level features by resize and average or concat
        gather_size = laterals[self.use_level].shape[2:]
        feats = []
        for i in range(self.num_outs):
            if i < self.use_level: 
                gathered = F.adaptive_max_pool2d(laterals[i], output_size=gather_size)
            else:
                gathered = F.interpolate(laterals[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        if self.ops == 'add':    
            feat = sum(feats) / len(feats)
        elif self.ops == 'concat':
            feat = torch.cat(feats,dim=1)
        else:
            res = self.fpn_conv(feats)
            return [res]

        feat = self.fpn_conv(feat)
        if self.use_down:
            assert self.use_level>=1
            siz = laterals[self.use_level-1].size()[2:]
            feat = self.last_conv(laterals[self.use_level-1]+F.interpolate(feat, size=siz, mode='nearest'))
        outs = []
        outs.append(feat)
        
        return tuple(outs)

if __name__ == "__main__":
    feats = []

    in_channels = [64,128,256,512]
    out_channels = 256
    num_outs = 2
    start_level=0
    end_level = -1
    use_level = 0
    ops = "concat"
    strides = [4,8,16,32]
    h,w = 1216,1600 
    for i in range(len(in_channels)):
        x = torch.randn(2,in_channels[i],h//strides[i],w//strides[i])
        feats.append(x)
       
    FPN = MyPAFPN(in_channels,out_channels,num_outs,start_level,end_leavel,use_level,ops)

    res = FPN(feats)

    