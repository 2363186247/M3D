from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import ITRDataset
from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer
import torch
from safetensors.torch import load_file
import os


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="./LaMed/pretrained_model/bert_base_uncased/")

    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    max_text_len: int = field(default=128)
    vocab_size: int = field(default=30522)


@dataclass
class DataArguments:
    data_root: str = field(default="./Data/data", metadata={"help": "Root directory for all data."})
    # caption data
    cap_data_path: str = field(default="./Data/data/M3D_Cap_npy/M3D_Cap.json", metadata={"help": "Path to caption data."})
    max_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    # config in bash file
    bf16: bool = True
    output_dir: str = "./LaMed/output/CLIP"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32 #32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04 # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    learning_rate: float = 1e-4 #1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all  # 是一个布尔值，表示是否在分布式训练中收集所有 GPU 的数据。

    def __call__(self, batch: list) -> dict:
        # 从批次中提取 image、text、input_id 和 attention_mask 字段，分别存储在 images、texts、input_ids 和 attention_mask 中。
        images, texts, input_ids, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask'))

        # 使用 unsqueeze(0) 将每个样本的张量扩展一个维度，然后使用 torch.cat 在第0维度上拼接成一个批次。
        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        # 计算批次大小。如果 gather_all 为 True，则获取分布式训练的总进程数，并将批次大小乘以总进程数。
        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)  # 生成一个从0到 batch_size-1 的标签张量。

        # 构建返回字典
        return_dict = dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,

        )

        return return_dict


def main():
    # 解析命令行参数，分为模型参数、数据参数和训练参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 从预训练模型路径加载 BertTokenizer 分词器
    tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)

    # 根据模型参数创建 M3DCLIPConfig
    config = M3DCLIPConfig.from_dict(vars(model_args))
    # 创建 M3DCLIP 模型
    model = M3DCLIP(config)

    # 如果指定了预训练模型，则加载预训练模型
    if model_args.pretrained_model:
        # 加载预训练模型权重
        # ckpt = torch.load(model_args.pretrained_model)
        ckpt = load_file(model_args.pretrained_model)
        # 将加载的权重加载到模型中，并设置 strict=True 确保权重严格匹配
        model.load_state_dict(ckpt, strict=True)
        # 打印加载预训练模型的信息
        print("load pretrained model.")

    # 创建训练数据集对象，传入数据参数、分词器和模式（'train'）
    train_dataset = ITRDataset(data_args, tokenizer, mode='train')
    # 创建验证数据集对象，传入数据参数、分词器和模式（'validation'）
    eval_dataset = ITRDataset(data_args, tokenizer, mode='validation')

    # 在每个 GPU 上独立计算 contrastive loss，并且将样本数据整理成模型可以接受的输入格式。
    # gather_all=False 表示不进行跨 GPU 的数据收集和损失计算。
    # 这通常用于资源有限或希望加快训练速度的情况。
    data_collator = DataCollator(False)

    # 创建 Trainer 对象，用于管理训练过程
    trainer = Trainer(
                        model=model,  # 指定要训练的模型
                        args=training_args,  # 传入训练参数
                        data_collator=data_collator,  # 传入数据整理器
                        train_dataset=train_dataset,  # 传入训练数据集
                        eval_dataset=eval_dataset,  # 传入验证数据集
                        compute_metrics=compute_metrics,  # 传入评估指标计算函数
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 传入 logits 预处理函数，用于计算评估指标
                      )

    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    trainer.train()

    # 保存训练状态
    trainer.save_state()
    # 保存模型配置
    model.config.save_pretrained(training_args.output_dir)
    # 保存模型
    model.save_pretrained(training_args.output_dir)
    # 保存分词器
    tokenizer.save_pretrained(training_args.output_dir)

    # 获取模型的状态字典
    state_dict = model.state_dict()
    # 将模型状态字典保存到文件
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":
    main()
