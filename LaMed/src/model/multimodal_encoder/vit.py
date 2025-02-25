# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,  # 输入图像的通道数
        img_size: Sequence[int] | int,  # 输入图像的尺寸（类型可以是一个整数序列（Sequence[int]）或一个整数（int））
        patch_size: Sequence[int] | int,  # 图像补丁的尺寸（类型可以是一个整数序列（Sequence[int]）或一个整数（int））
        hidden_size: int = 768,  # 隐藏层的大小
        mlp_dim: int = 3072,  # 多层感知机的维度
        num_layers: int = 12,  # 层数
        num_heads: int = 12,  # 注意力头的数量
        pos_embed: str = "conv",  # 位置嵌入的类型
        classification: bool = False,  # 是否使用分类
        num_classes: int = 2,  # 分类的类别数
        dropout_rate: float = 0.0,  # dropout的比例
        spatial_dims: int = 3,  # 空间维度
        post_activation="Tanh",  # 分类头的激活函数
        qkv_bias: bool = False,  # 是否在自注意力块中应用偏置
        save_attn: bool = False,  # 是否保存自注意力块中的注意力
    ) -> None:  # 表示函数不返回任何值（即返回 None）
        """
        Args:
            in_channels (int): 输入图像的通道数
            img_size (Union[Sequence[int], int]): 输入图像的尺寸
            patch_size (Union[Sequence[int], int]): 图像补丁的尺寸
            hidden_size (int, optional): 隐藏层的大小，默认为768
            mlp_dim (int, optional): 多层感知机的维度，默认为3072
            num_layers (int, optional): 层数，默认为12
            num_heads (int, optional): 注意力头的数量，默认为12
            pos_embed (str, optional): 位置嵌入的类型，默认为"conv"
            classification (bool, optional): 是否使用分类，默认为False
            num_classes (int, optional): 分类的类别数，默认为2
            dropout_rate (float, optional): dropout的比例，默认为0.0
            spatial_dims (int, optional): 空间维度，默认为3
            post_activation (str, optional): 分类头的激活函数，默认为"Tanh"
            qkv_bias (bool, optional): 是否在自注意力块中应用偏置，默认为False
            save_attn (bool, optional): 是否保存自注意力块中的注意力，默认为False
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):  # 检查dropout_rate是否在0到1之间
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:  # 检查hidden_size是否能被num_heads整除
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size  # 隐藏层的大小（768）
        self.classification = classification  # 是否使用分类
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,  # 输入图像的通道数（1）
            img_size=img_size,  # 输入图像的尺寸（32,256,256）
            patch_size=patch_size,  # 图像补丁的尺寸（4,16,16）
            hidden_size=hidden_size,  # 隐藏层的大小（768）
            num_heads=num_heads,  # 注意力头的数量（12）
            pos_embed=pos_embed,  # 位置嵌入的类型（'perceptron'）
            dropout_rate=dropout_rate,  # dropout的比例(0.0)
            spatial_dims=spatial_dims,  # 空间维度(3)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)  # 创建多个Transformer块
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)  # 层归一化
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))  # 分类token
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        x = self.patch_embedding(x)  # 图像补丁嵌入（将图像分割成patch，然后将每个patch转换为一个向量表示）
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展分类token
            x = torch.cat((cls_token, x), dim=1)  # 拼接分类token和图像补丁
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)  # 通过Transformer块
            hidden_states_out.append(x)  # 保存每层的输出
        x = self.norm(x)  # 层归一化
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])  # 分类头
        return x, hidden_states_out  # 返回最后一层的输出和所有层的隐藏状态

class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 配置
        self.select_layer = config.vision_select_layer  # 选择的层
        self.select_feature = config.vision_select_feature  # 选择的特征

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,  # 输入图像的通道数
            img_size=self.config.image_size,  # 输入图像的尺寸
            patch_size=self.config.patch_size,  # 图像补丁的尺寸
            pos_embed="perceptron",  # 位置嵌入的类型
            spatial_dims=len(self.config.patch_size),  # 空间维度
            classification=True,  # 是否使用分类
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)  # 获取最后一层的输出和所有层的隐藏状态
        if self.select_layer == -1:
            image_features = last_feature  # 选择最后一层的输出
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]  # 选择特定层的输出
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # 选择patch特征，即去掉第一个分类 token。
        elif self.select_feature == 'cls_patch':
            image_features = image_features  # 选择分类和patch特征，即保留所有特征。
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features  # 返回图像特征

    @property
    def dtype(self):
        return self.vision_tower.dtype  # 返回数据类型

    @property
    def device(self):
        return self.vision_tower.device  # 返回设备

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size  # 返回隐藏层的大小