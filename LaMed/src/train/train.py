import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})



@dataclass
class DataArguments:
    data_root: str = field(default="./Data/data/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="./Data/data/M3D_Cap_npy/M3D_Cap.json", metadata={"help": "Path to caption data."})

    # VQA data
    vqa_data_train_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_train.csv", metadata={"help": "Path to training VQA data."})
    vqa_data_val_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_val.csv", metadata={"help": "Path to validation VQA data."})
    vqa_data_test_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_test.csv", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_yn_train.csv", metadata={"help": "Path to training VQA Yes or No data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="./Data/data/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
    refseg_data_train_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16            # 秩（rank）
    lora_alpha: int = 32        # 缩放因子（用于调整低秩适配器的输出幅度，以确保其与原始模型的输出在同一数量级上。）
    lora_dropout: float = 0.05  # dropout 率
    lora_weight_path: str = ""  # 权重路径
    lora_bias: str = "none"     # 偏置配置

    cache_dir: Optional[str] = field(default=None)  # 缓存目录
    remove_unused_columns: bool = field(default=False)  # 是否移除未使用的列
        # 在使用 transformers 库进行模型训练时，数据集通常包含多个列（字段），但并不是所有的列都会被模型使用。
        # 例如，一个数据集可能包含文本、标签、元数据等多个列，而模型训练过程中可能只需要使用文本和标签列。

    model_max_length: int = field(
        default=512, # 最大序列长度512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42  # 随机种子
    ddp_backend: Optional[str] = None   # 禁用分布式数据并行（DDP）
    # ddp_backend: str = "nccl"   # 分布式数据并行（DDP）的后端（NVIDIA Collective Communications Library）
    ddp_timeout: int = 128000   # DDP 的超时时间
    ddp_find_unused_parameters: bool = False    # 是否查找未使用的参数
    optim: str = field(default="adamw_torch")   # 优化器类型

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            images, input_ids, labels, attention_mask = tuple(
                [b[key] for b in batch if b is not None] for key in ('image', 'input_id', 'label', 'attention_mask'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        return return_dict


def main():
    global local_rank  # 用于存储当前进程的本地排名（在分布式训练中使用）。如果没有使用分布式训练，则为None。
    
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # 填充的方向
        use_fast=False,  # 不使用快速分词器
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif 'phi3' in model_args.model_type:
            model = LamedPhi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
                )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    # 如果 freeze_backbone 参数为 True，则冻结模型的主干网络，使其不参与梯度计算。
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()  # 启用输入的梯度计算
    
    # 启用梯度检查点功能。梯度检查点是一种节省显存的方法，
    # 通过在前向传播过程中保存部分中间结果，减少反向传播时的显存占用。
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)  # 冻结模型的所有参数（即不计算梯度）
        
        # 解冻特定模块的参数。
        # 多模态投影模块
        # 用于将不同模态的特征投影到一个共同的特征空间中，以便进行联合处理和学习。
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # 检查是否提供了预训练的 MLLM 权重路径
    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")  # 权重被加载到 CPU 上，避免在加载过程中占用 GPU 内存。
        model.load_state_dict(ckpt, strict=True)  # strict=True 确保权重文件中的所有参数都能匹配到模型中的参数
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        # 配置 LoRA 适配器
        lora_config = LoraConfig(
            r=training_args.lora_r,                         # 秩（rank）
            lora_alpha=training_args.lora_alpha,            # 缩放因子
            target_modules=find_all_linear_names(model),    # 目标模块（使用 find_all_linear_names 找到模型中的所有线性层）
            lora_dropout=training_args.lora_dropout,        # dropout 率
            bias=training_args.lora_bias,                   # 偏置配置
            task_type="CAUSAL_LM",                          # 任务类型
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        # 遍历模型的所有参数，并检查参数名称是否包含特定关键词（如 vision_tower、mm_projector 等）。
        # 如果包含这些关键词，则将参数的 requires_grad 属性设置为 True，即这些参数在训练过程中是可训练的。
        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    # ckpt = torch.load("PATH/model_with_lora.bin", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")  # 检查模型是否包含 seg_module 模块

    if model_args.tune_mm_mlp_adapter:
        train_dataset = TextDatasets(data_args, tokenizer, mode='train')
    else:
        train_dataset = UniDatasets(data_args, tokenizer, mode='train')

    eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable)

    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,  # 数据整理器，用于在训练过程中整理和批处理数据
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,  # 用于计算评估指标的函数
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics  # 用于在计算评估指标之前预处理模型输出的函数
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True  # 用于在推理阶段启用缓存，以提高推理速度。

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()  # 取模型的状态字典 state_dict ，其中包含了模型的所有参数和缓冲区。
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))  # 将状态字典保存到指定路径
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)  # 保存普通模型的状态字典


if __name__ == "__main__":
    main()
