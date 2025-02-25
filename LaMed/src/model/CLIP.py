import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.utils.dist_utils import gather_features


class M3DCLIPConfig(PretrainedConfig):
    model_type = "m3d_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",  # 语言模型的路径或名称
        local_loss: bool = False,  # 是否使用局部损失
        gather_loss: bool = True,  # 是否使用全局损失
        in_channels: int = 1,  # 输入图像的通道数
        img_size: tuple = (32, 256, 256),  # 输入图像的尺寸
        patch_size: tuple = (4, 16, 16),  # 图像补丁的尺寸
        hidden_size: int = 768,  # 隐藏层的大小
        mlp_dim: int = 3072,  # 多层感知机的维度
        num_layers: int = 12,  # 层数
        num_heads: int = 12,  # 注意力头的数量
        pos_embed: str = "perceptron",  # 位置嵌入的类型
        dropout_rate: float = 0,  # dropout的比例
        spatial_dims: int = 3,  # 空间维度
        max_text_len: int = 128,  # 文本的最大长度
        vocab_size: int = 30522,  # 词汇表的大小
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path  # 语言模型的路径或名称
        self.in_channels = in_channels  # 输入图像的通道数（1）
        self.img_size = img_size  # 输入图像的尺寸（32,256,256）
        self.patch_size = patch_size  # 图像补丁的尺寸（4,16,16）
        self.hidden_size = hidden_size  # 隐藏层的大小（768）
        self.mlp_dim = mlp_dim  # 多层感知机的维度（3072）
        self.num_layers = num_layers  # 层数（12）
        self.num_heads = num_heads  # 注意力头的数量（12）
        self.pos_embed = pos_embed  # 位置嵌入的类型（perceptron）
        self.dropout_rate = dropout_rate  # dropout的比例（0.0）
        self.spatial_dims = spatial_dims  # 空间维度（3）
        self.local_loss = local_loss  # 是否使用局部损失（False）
        self.gather_loss = gather_loss  # 是否使用全局损失
        self.max_text_len = max_text_len  # 文本的最大长度（128）
        self.vocab_size = vocab_size  # 词汇表的大小（30522）
        super().__init__(**kwargs)


class M3DCLIP(PreTrainedModel):
    config_class = M3DCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = ViT(
            in_channels=config.in_channels,  # 输入图像的通道数（1）
            img_size=config.img_size,  # 输入图像的尺寸（32,256,256）
            patch_size=config.patch_size,  # 图像补丁的尺寸（4,16,16）
            hidden_size=config.hidden_size,  # 隐藏层的大小（768）
            mlp_dim=config.mlp_dim,  # 多层感知机的维度（3072）
            num_layers=config.num_layers,  # 层数（12）
            num_heads=config.num_heads,  # 注意力头的数量（12）
            pos_embed=config.pos_embed,  # 位置嵌入的类型（perceptron）
            dropout_rate=config.dropout_rate,  # dropout的比例（0.0）
            spatial_dims=config.spatial_dims,  # 空间维度（3）
            classification=True,
        )

        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)  # 语言编码器

        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size)  # 视觉特征投影
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)  # 语言特征投影

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # logit缩放参数

        self.local_loss = config.local_loss  # 是否使用局部损失
        self.gather_loss = config.gather_loss  # 是否使用全局损失

    def encode_image(self, image):
        image_feats, _ = self.vision_encoder(image)  # 编码图像
        image_feats = self.mm_vision_proj(image_feats)  # 视觉特征投影
        image_feats = F.normalize(image_feats, dim=-1)  # 归一化

        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]  # 编码文本
        text_feats = self.mm_language_proj(text_feats)  # 语言特征投影
        text_feats = F.normalize(text_feats, dim=-1)  # 归一化

        return text_feats

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features = self.encode_image(images)[:, 0]  # 获取图像特征（取编码后的特征的第一个元素）
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]  # 获取文本特征（取编码后的特征的第一个元素）

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)  # 聚合特征
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T  # 图像logits
                logits_per_text = self.logit_scale * text_features @ all_image_features.T  # 文本logits
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T  # 图像logits
                logits_per_text = logits_per_image.T  # 文本logits
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T  # 图像logits
            logits_per_text = self.logit_scale * text_features @ image_features.T  # 文本logits

        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2  # 计算损失

        ret = {
            "loss": loss,  # 损失
            "logits": (logits_per_image + logits_per_text) / 2.0,  # logits
        }

        return ret

# 注册自定义配置类和模型类，以便在使用 AutoConfig 和 AutoModel 时能够自动识别和加载它们。
AutoConfig.register("m3d_clip", M3DCLIPConfig)
AutoModel.register(M3DCLIPConfig, M3DCLIP)