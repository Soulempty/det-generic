import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS

class SPP(nn.Module):
    def __init__(self, c, k=(3, 5)):
        super(SPP, self).__init__()
        self.cv2 = ConvModule(c * (len(k) + 1), c, 1, norm_cfg=dict(type='BN', requires_grad=True),inplace=False)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@NECKS.register_module()
class AugFPN(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_spp = False):
        super(AugFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels  # [256, 512, 1024, 2048]
        self.out_channels = out_channels  #256
        self.num_ins = len(in_channels)
        self.num_outs = num_outs  #5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.use_spp = use_spp

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_dim = out_channels if i==0 else out_channels + in_channels[self.start_level]
            l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            
            fpn_conv = ConvModule(
                in_dim,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        if use_spp:
            self.spp = SPP(in_channels[self.start_level])

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
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = inputs[i + self.start_level]
            if i == 0 and self.use_spp:
                x = self.spp(x)
            y = lateral_conv(x) 
            laterals.append(y)
        # build top-down path
        used_backbone_levels = len(laterals) #4   321
        
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = []
        low_feat = inputs[self.start_level]
        for i in range(used_backbone_levels): #0123
            x = laterals[i]
            if i > 0:
                x = torch.cat([x,F.adaptive_max_pool2d(low_feat, output_size=x.shape[2:])],dim=1)
            outs.append(self.fpn_convs[i](x))
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))  
        return tuple(outs)

@NECKS.register_module()
class BupFPN(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_spp = False):
        super(BupFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels  # [256, 512, 1024, 2048]
        self.out_channels = out_channels  #256
        self.num_ins = len(in_channels)
        self.num_outs = num_outs  #5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.use_spp = use_spp

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            in_dim = out_channels if i==0 else out_channels + in_channels[self.start_level]
            l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            
            fpn_conv = ConvModule(
                in_dim,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        if use_spp:
            self.spp = SPP(in_channels[self.start_level])

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
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = inputs[i + self.start_level]
            if i == 0 and self.use_spp:
                x = self.spp(x)
            y = lateral_conv(x) 
            laterals.append(y)
        # build top-down path
        used_backbone_levels = len(laterals) #4   321
        
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = []
        low_feat = inputs[self.start_level]
        for i in range(used_backbone_levels): #0123
            x = laterals[i]
            if i > 0:
                x = torch.cat([x,F.adaptive_max_pool2d(low_feat, output_size=x.shape[2:])],dim=1)
            outs.append(self.fpn_convs[i](x))
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))  
        return tuple(outs)