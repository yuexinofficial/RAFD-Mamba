# -*- coding: utf-8 -*-
"""模块 B3 = EccoMamba 的 **HAE**(论文名 Hierarchical Aggregation Enhancement)。

来源:``models/eccomamba/vmamba.py`` 里的 ``MSC2`` 类,**逐字复制**。
原作者仓库 Biowust/EccoMamba,该文件与我们这边逐字节相同。
本段代码 md5 = 13676f95742a13a93e77947c8538ae5b
**除本段说明外一字未改。**

EccoMamba 论文对 HAE 的描述:
    "in the encoder's downsampling path, we introduce a Hierarchical Aggregation
     Enhancement (HAE) module that integrates multi-scale convolutions with
     hierarchical attention mechanisms."

挂载位置(与 EccoMamba 一致):**编码器每个 stage 的入口**,在 VSS block 之前,
形式是 ``x = x + self.msc(x)``。EccoMamba 在全部 4 个 stage 都挂 ⇒ 这里也是 4 个。

🔴 与 A / B1 / B2 的一个关键差别:**MSC2 是非零初始化的。**
   它的 ``proj``(Conv2d(2C, C, 1)) 用 PyTorch 默认初始化,``conv_fuse`` 也非零
   ⇒ **挂上就改变初始前向**,没有"训练前逐位恒等"这个性质。
   所以本臂与 ``骨干 + A1(4级)`` 的差值里,**混了"多一个模块"和"初始前向被扰动了"两件事**。
   (A/RAFR、B1/DMSC、B2/SLGE 都是零初始化输出投影,只有这一个不是。)

参数量(dim = 96/192/384/768):28,514 + 108,194 + 421,346 + 1,662,818 = **2,220,872**。
结构:输入按通道劈成 1/2 : 1/4 : 1/4 三份,分别走 MaxPool / (通道+空间)注意力 / 深度可分离卷积,
     各自 1×1 升维后 concat,再 ``x + dwconv3x3(x)``,最后 1×1 投回原通道,并残差回输入。

tensor 布局:与 DMSC 一致 —— **BHWC 进 BHWC 出**(函数开头/结尾各 permute 一次)。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class MSC2(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, proj_drop=0.,
        **kwargs, ):
        super().__init__()
        
        # maxpool占1/2的输入，attention占1/4的输入，dwconv占1/4的输入
        self.maxpool_in = maxpool_in = dim // 2 # 1/2 of original input (since dim is already X/2)
        self.attention_in = attention_in = dim // 4  # 1/8 of original input
        self.dwconv_in = dwconv_in = dim // 4  # 1/8 of original input
        
        self.maxpool_dim = maxpool_dim = maxpool_in * 2
        self.attention_dim = attention_dim = attention_in * 2
        self.dwconv_dim = dwconv_dim = dwconv_in * 2

        # MaxPool分支
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj_maxpool = nn.Conv2d(maxpool_in, maxpool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu_maxpool = nn.GELU()

        # Channel + Spatial Attention分支
        self.attention_conv = nn.Conv2d(attention_in, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.attention_proj = nn.Conv2d(attention_dim, attention_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=attention_dim)
        self.mid_gelu_attention = nn.GELU()
        
        # 集成通道注意力和空间注意力
        self.channel_attention = self._build_channel_attention(attention_dim)
        self.spatial_attention = self._build_spatial_attention()

        # DWconv分支
        self.dwconv_conv = nn.Conv2d(dwconv_in, dwconv_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwconv_proj = nn.Conv2d(dwconv_dim, dwconv_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=dwconv_dim)
        self.mid_gelu_dwconv = nn.GELU()

        # 融合层
        total_dim = maxpool_dim + attention_dim + dwconv_dim
        self.conv_fuse = nn.Conv2d(total_dim, total_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=total_dim)
        self.proj = nn.Conv2d(total_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_channel_attention(self, in_planes, ratio=16):
        """构建通道注意力模块"""
        class ChannelAttentionModule(nn.Module):
            def __init__(self, in_planes, ratio):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # 输入已经是 B C H W 格式
                avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
                max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
                out = avg_out + max_out
                return self.sigmoid(out)
        
        return ChannelAttentionModule(in_planes, ratio)

    def _build_spatial_attention(self, kernel_size=7):
        """构建空间注意力模块"""
        class SpatialAttentionModule(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
                padding = 3 if kernel_size == 7 else 1
                self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # 输入已经是 B C H W 格式
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                x = self.conv1(x)
                return self.sigmoid(x)
        
        return SpatialAttentionModule(kernel_size)
        
    def forward(self, x):
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2)
        # MaxPool分支 - 使用全部输入
        maxpool_x = x[:, :self.maxpool_in, :, :].contiguous()  # 使用全部输入 (1/2 of original)
        maxpool_x = self.Maxpool(maxpool_x)
        maxpool_x = self.proj_maxpool(maxpool_x)
        maxpool_x = self.mid_gelu_maxpool(maxpool_x)
        
        # Channel + Spatial Attention分支 - 使用1/4输入  
        attention_x = x[:, self.maxpool_in:self.maxpool_in+self.attention_in, :, :].contiguous()  # 取前1/4通道
        attention_x = self.attention_conv(attention_x)
        attention_x = self.attention_proj(attention_x)
        attention_x = self.mid_gelu_attention(attention_x)
        
        # 应用通道注意力
        ca_weight = self.channel_attention(attention_x)
        attention_x = attention_x * ca_weight
        
        # 应用空间注意力
        sa_weight = self.spatial_attention(attention_x)
        attention_x = attention_x * sa_weight
        # DWconv分支 - 使用1/4输入
        dwconv_x = x[:, self.attention_in+self.maxpool_in:, :, :].contiguous()  # 取接下来1/4通道
        dwconv_x = self.dwconv_conv(dwconv_x)
        dwconv_x = self.dwconv_proj(dwconv_x)
        dwconv_x = self.mid_gelu_dwconv(dwconv_x)

        # 拼接三个分支
        x = torch.cat((maxpool_x, attention_x, dwconv_x), dim=1)

        # 融合和投影
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        return x

    