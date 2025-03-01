from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"): # 检查 config 是否具有 vision_tower 属性。
            self.vision_tower = build_vision_tower(config)  # 根据 config 构建 vision_tower。
            self.mm_projector = build_mm_projector(config)  # 根据 config 构建 mm_projector。

        # 如果 config 具有 segmentation_module 属性且不为 None，则构建 segmentation_module。
        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config) # 根据 config 构建 seg_module。

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()   # 初始化二值 Dice 损失函数
            self.bce_loss = BCELoss()           # 初始化二值交叉熵损失函数

    def get_vision_tower(self):
        # 如果实例中存在 vision_tower 属性，则返回该属性的值；如果不存在，则返回 None。
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
            # 指定在 vision tower 中使用哪一层的输出特征。
            # vision tower 通常由多个层组成，每一层提取不同层次的特征。
            # 通过指定 vision_select_layer，可以选择最适合当前任务的特征层。

        self.config.vision_select_feature = model_args.vision_select_feature
            # 指定在 vision tower 中选择的特征类型。
            # 不同的层可能会输出不同类型的特征，例如卷积特征、池化特征等。
            # 通过指定 vision_select_feature，可以选择最适合当前任务的特征类型。

        self.config.mm_projector_type = model_args.mm_projector_type
            # 指定多模态投影器的类型，用于将视觉特征投影到多模态特征空间中。
            # 不同的投影器类型可能使用不同的架构或方法来实现特征投影。

        self.config.proj_layer_type = model_args.proj_layer_type
            # 指定投影层的类型，用于定义多模态投影器中的层类型。
            # 例如，可以是线性层、卷积层等。

        self.config.proj_layer_num = model_args.proj_layer_num
            # 指定多模态投影器中的层数量。
            # 通过设置层的数量，可以控制投影器的深度和复杂度。

        self.config.proj_pooling_type = model_args.proj_pooling_type
            # 指定多模态投影器中的池化类型。
            # 池化操作用于减少特征图的尺寸，同时保留重要的特征信息。

        self.config.proj_pooling_size = model_args.proj_pooling_size
            # 指定多模态投影器中的池化大小。
            # 池化大小决定了池化操作的窗口大小，从而影响池化后的特征图尺寸。

        # vision tower
        if self.get_vision_tower() is None: # 如果实例中不存在 vision_tower，则构建 vision_tower。
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)
                # freeze_vision_tower 为 True，则 requires_grad_ 设置为 False，冻结 vision tower 的参数。
                # freeze_vision_tower 为 False，则 requires_grad_ 设置为 True，允许 vision tower 的参数在训练过程中更新。


        if model_args.pretrain_vision_model is not None:    # 如果指定了预训练的视觉模型，则加载预训练的视觉模型。
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            
            # 将加载的预训练权重 vision_model_weights 加载到 vision_tower 的模型中。
            # strict=True 表示严格匹配模型的结构和权重。
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)

        # 确保 mm_projector 的隐藏层大小与 vision_tower 的隐藏层大小一致
        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None: # 如果实例中不存在 mm_projector，则构建 mm_projector。
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:  # 如果指定了预训练的多模态 MLP 适配器，则加载预训练的多模态 MLP 适配器。
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                # weights.items() ：返回权重字典中的所有键值对。
                # if keyword in k ：过滤出键中包含 keyword 的键值对。
                # k.split(keyword + '.')[1] ：将键按 keyword + '.' 分割，并取分割后的第二部分作为新的键。
                # v 保持原来的值不变。

                # 示例：
                # weights = {
                #     'mm_projector.layer1.weight': torch.tensor([1, 2, 3]),
                #     'mm_projector.layer1.bias': torch.tensor([4, 5, 6]),
                #     'mm_projector.layer2.weight': torch.tensor([7, 8, 9]),
                #     'other_module.layer1.weight': torch.tensor([10, 11, 12])
                # }
                # keyword = 'mm_projector'
                # filtered_weights = get_w(weights, keyword)
                # 调用 get_w 函数后，filtered_weights 将包含以下键值对：
                # filtered_weights = {
                #     'layer1.weight': torch.tensor([1, 2, 3]),
                #     'layer1.bias': torch.tensor([4, 5, 6]),
                #     'layer2.weight': torch.tensor([7, 8, 9])
                # }

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:   # 如果实例中不存在 seg_module，则构建 seg_module。
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True  # 启用 seg_module

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue    # 跳过 key 中以 'model.text_encoder.' 或 'text_encoder.' 开头的键值对
                if key.startswith('model.'):
                    new_key = key[len('model.'):]       # 去掉 key 中的 'model.' 前缀
                    new_state_dict[new_key] = value     # 将去掉前缀的 key 和 value 添加到 new_state_dict 中
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()   # 初始化二值 Dice 损失函数
        self.bce_loss = BCELoss()           # 初始化二值交叉熵损失函数

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
                # inputs_embeds[:, :1, :]：
                # 作用：提取输入嵌入的第一个标识符嵌入。
                # 原因：保留输入序列的起始标识符，通常是特殊的起始标记。
                
                # image_features：
                # 作用：包含图像的特征表示。
                # 原因：将图像特征插入到输入嵌入中，以便模型能够处理多模态输入。

                # inputs_embeds[:, (image_features.shape[1] + 1):, :]：
                # 作用：提取剩余的输入嵌入，跳过图像特征长度的部分。
                # 原因：确保图像特征插入后，保留输入序列的其余部分。

                # 实现将图像特征和文本输入嵌入结合起来，形成多模态输入。
                # 确保输入序列的起始标识符和图像特征都被包含在最终的输入嵌入中。
                # 同时保留了输入序列的其余部分。

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))    # 调整模型的嵌入层大小，以适应新的 token 数量。

        if num_new_tokens > 0:  # 如果存在新的 token
            input_embeddings = self.get_input_embeddings().weight.data      # 获取输入嵌入的权重数据
            output_embeddings = self.get_output_embeddings().weight.data    # 获取输出嵌入的权重数据

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)    # 从输入嵌入中排除新添加的 token 后，计算输入嵌入的平均值
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)    # 从输出嵌入中排除新添加的 token 后，计算输出嵌入的平均值

            input_embeddings[-num_new_tokens:] = input_embeddings_avg   # 将输入嵌入的新 token 部分设置为输入嵌入的平均值
            output_embeddings[-num_new_tokens:] = output_embeddings_avg # 将输出嵌入的新 token 部分设置为输出嵌入的平均值

            if model_args.tune_mm_mlp_adapter:  # 如果需要调整多模态 MLP 适配器
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True  # 允许输入嵌入的参数在训练过程中更新
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False # 禁止输出嵌入的参数在训练过程中更新
            else:   # 如果不需要调整多模态 MLP 适配器
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True  # 允许输入嵌入的参数在训练过程中更新
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True  # 允许输出嵌入的参数在训练过程中更新

        if model_args.pretrain_mm_mlp_adapter:  # 如果指定了预训练的多模态 MLP 适配器
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")