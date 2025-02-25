import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates
from .term_dictionary import term_dict



class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args  # 保存传入的参数
        self.data_root = args.data_root  # 数据根目录('./Data/data')
        self.tokenizer = tokenizer  # 分词器
        self.mode = mode  # 模式（train/validation/test）

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)  # 加载数据文件
        self.data_list = self.json_file[mode]  # 获取对应模式的数据列表

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                    # spatial_axes 是一个元组，用于指定多维图像数据的旋转轴。
                    # 在三维图像数据中，通常有三个轴：深度、高度、宽度。

                mtf.RandFlip(prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=2),  # 随机翻转
                    # spatial_axis 是一个整数，用于指定单个轴上的翻转操作。

                mtf.RandScaleIntensity(factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
            self.data_list = self.data_list[:512]  # 验证模式只使用前512个数据
        elif 'test' in mode:
            self.transform = val_transform  # 测试模式使用验证变换

    def __len__(self):
        return len(self.data_list)

    # 将输入文本 input_text 截断到指定的最大标记数 max_tokens。
    # 通过分词器对文本进行编码，并计算编码后的标记数量。
    # 如果标记数量超过了最大标记数，则通过选择和组合句子来截断文本。
    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            # 使用分词器对文本进行编码，并计算编码后的标记数量
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        # 如果输入文本的标记数量小于或等于最大标记数，则直接返回输入文本
        if count_tokens(input_text) <= max_tokens:
            return input_text

        # 将输入文本按句号分割成多个句子
        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        # 如果存在句子，则将第一个句子添加到选定句子列表中
        if sentences:
            selected_sentences.append(sentences.pop(0))

        # 若当前标记数量小于或等于最大标记数且存在剩余句子时，随机选择一个句子
        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            # 如果添加该句子后标记数量仍小于或等于最大标记数且该句子不在选定句子列表中，则将其添加到选定句子列表中
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        # 将选定的句子重新组合成一个字符串并返回
        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx] # 从数据列表中获取索引为 idx 的数据项。
                image_path = data["image"] # 获取图像路径。
                image_abs_path = os.path.join(self.data_root, image_path) # 构建图像的绝对路径。

                # 加载图像数据，假设图像数据已经归一化到0-1范围。
                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image) # 对图像应用预定义的变换。

                text_path = data["text"] # 获取文本路径。
                text_abs_path = os.path.join(self.data_root, text_path) # 构建文本的绝对路径。
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read() # 读取文本内容。

                # 调用 truncate_text 方法，将文本截断到指定的最大标记数。
                text = self.truncate_text(raw_text, self.args.max_length)

                # 使用分词器对文本进行编码，并将其转换为PyTorch张量。
                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                # 获取编码后的输入ID和注意力掩码。
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                # 构建包含图像、文本、输入ID、注意力掩码和问题类型的字典。
                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1) # 如果出现异常，则随机选择一个新的索引。




class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args  # 保存传入的参数
        self.data_root = args.data_root  # 数据根目录
        self.tokenizer = tokenizer  # 分词器
        self.mode = mode  # 模式（train/validation/test）

        self.image_tokens = "<im_patch>" * args.proj_out_num  # 图像标记（将 <im_patch> 重复 args.proj_out_num 次）

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)  # 加载数据文件
        self.data_list = self.json_file[mode]  # 获取对应模式的数据列表

        self.caption_prompts = Caption_templates  # Caption提示

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                mtf.RandFlip(prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=2),  # 随机翻转
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
        elif 'test' in mode:
            self.transform = val_transform  # 测试模式使用验证变换

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                # 获取数据列表中的数据
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                # 加载图像数据并进行预处理
                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                # 加载文本数据
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                # 随机选择一个标题模板
                prompt_question = random.choice(self.caption_prompts)

                # 构建问题
                question = self.image_tokens + prompt_question

                # 使用分词器对文本进行编码，并将其转换为PyTorch张量
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                # 获取编码后的输入ID和注意力掩码
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask) #  计算 attention_mask 中所有非零元素的和，即有效标记的数量。
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, 
                    max_length=self.args.max_length, 
                    truncation=True, # 如果文本长度超过最大长度，则进行截断。
                    padding="max_length", 
                    return_tensors="pt" # 返回 PyTorch 张量。
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                # 构建包含图像、输入ID、标签、注意力掩码、问题和答案的字典
                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                # 如果在获取数据时发生异常，则打印错误信息并随机选择一个新的索引重新尝试
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args  # 保存传入的参数
        self.data_root = args.data_root  # 数据根目录
        self.tokenizer = tokenizer  # 分词器
        self.mode = mode  # 模式（train/validation/test）
        self.close_ended = close_ended  # 是否为封闭式问题

        self.image_tokens = "<im_patch>" * args.proj_out_num  # 图像标记（将 <im_patch> 重复 args.proj_out_num 次）

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)  # 加载训练数据
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)  # 加载验证数据
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)  # 加载测试数据
        else:
            print("The mode is not desired ! ")

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                mtf.RandFlip(prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=2),  # 随机翻转
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
        elif 'test' in mode:
            self.transform = val_transform  # 测试模式使用验证变换

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]  # 获取数据列表中的数据
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])  # 构建图像的绝对路径

                image = np.load(image_abs_path)  # 加载图像数据并进行预处理
                image = self.transform(image)  # 对图像应用预定义的变换

                # 处理封闭式问题
                if self.close_ended:
                    question = data["Question"]  # 获取问题
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    question = question + ' ' + choices  # 构建封闭式问题
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])  # 获取答案
                # 处理开放式问题
                else:
                    question = data["Question"]  # 获取问题
                    answer = str(data["Answer"])  # 获取答案

                question = self.image_tokens + ' ' + question  # 构建问题
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对文本进行编码，并将其转换为PyTorch张量

                input_id = text_tensor["input_ids"][0]  # 获取编码后的输入ID
                attention_mask = text_tensor["attention_mask"][0]  # 获取注意力掩码

                valid_len = torch.sum(attention_mask)  # 计算 attention_mask 中所有非零元素的和，即有效标记的数量
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于输入ID长度，则在有效标记位置添加结束标记

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对问题进行编码，并将其转换为PyTorch张量
                question_len = torch.sum(question_tensor["attention_mask"][0])  # 计算问题的有效标记数量

                label = input_id.clone()  # 复制输入ID作为标签
                label[:question_len] = -100  # 将问题部分的标签设置为-100，表示不计算损失
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于标签长度，则在有效标记位置添加结束标记
                else:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100

                ret = {
                    'image': image,  # 图像数据
                    'input_id': input_id,  # 输入ID
                    'label': label,  # 标签
                    'attention_mask': attention_mask,  # 注意力掩码
                    'question': question,  # 问题
                    'answer': answer,  # 答案
                    'answer_choice': data["Answer Choice"],  # 答案选择
                    'question_type': data["Question Type"],  # 问题类型
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})  # 如果启用分割，则添加分割数据

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)  # 如果出现异常，则随机选择一个新的索引


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args  # 保存传入的参数
        self.data_root = args.data_root  # 数据根目录
        self.tokenizer = tokenizer  # 分词器
        self.mode = mode  # 模式（train/validation/test）

        self.image_tokens = "<im_patch>" * args.proj_out_num  # 图像标记（将 <im_patch> 重复 args.proj_out_num 次）

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)  # 加载训练数据
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)  # 加载验证数据
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)  # 加载测试数据
        else:
            print("The mode is not desired ! ")

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                mtf.RandFlip(prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlip(prob=0.10, spatial_axis=2),  # 随机翻转
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
        elif 'test' in mode:
            self.transform = val_transform  # 测试模式使用验证变换

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]  # 获取数据列表中的数据
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])  # 构建图像的绝对路径

                image = np.load(image_abs_path)  # 加载图像数据并进行预处理
                image = self.transform(image)  # 对图像应用预定义的变换

                question = data["Question"]  # 获取问题
                answer = str(data["Answer"])  # 获取答案

                question = self.image_tokens + ' ' + question  # 构建问题
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对文本进行编码，并将其转换为PyTorch张量

                input_id = text_tensor["input_ids"][0]  # 获取编码后的输入ID
                attention_mask = text_tensor["attention_mask"][0]  # 获取注意力掩码

                valid_len = torch.sum(attention_mask)  # 计算 attention_mask 中所有非零元素的和，即有效标记的数量
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于输入ID长度，则在有效标记位置添加结束标记

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对问题进行编码，并将其转换为PyTorch张量
                question_len = torch.sum(question_tensor["attention_mask"][0])  # 计算问题的有效标记数量

                label = input_id.clone()  # 复制输入ID作为标签
                label[:question_len] = -100  # 将问题部分的标签设置为-100，表示不计算损失
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于标签长度，则在有效标记位置添加结束标记
                else:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100

                ret = {
                    'image': image,  # 图像数据
                    'input_id': input_id,  # 输入ID
                    'label': label,  # 标签
                    'attention_mask': attention_mask,  # 注意力掩码
                    'question': question,  # 问题
                    'answer': answer,  # 答案
                    'answer_choice': data["Answer Choice"],  # 答案选择
                    'question_type': data["Question Type"],  # 问题类型
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})  # 如果启用分割，则添加分割数据

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)  # 如果出现异常，则随机选择一个新的索引


class PosRECDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args  # 保存传入的参数
        self.tokenizer = tokenizer  # 分词器

        self.tag = tag  # 数据集标签
        self.mode = mode  # 模式（train/validation/test）
        self.description = description  # 是否使用描述

        self.dataset_info = dataset_info  # 数据集信息

        self.image_tokens = "<im_patch>" * args.proj_out_num  # 图像标记（将 <im_patch> 重复 args.proj_out_num 次）
        self.box_tokens = ["<bx_start>", "<bx_end>"]  # 框标记

        root_path = args.seg_data_path  # 分割数据路径
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )  # 加载训练数据
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )  # 加载验证数据
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )  # 加载测试数据

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),  # 随机翻转
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensord(keys=["image"], dtype=torch.float),  # 转换为张量
                mtf.ToTensord(keys=["seg"], dtype=torch.int),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),  # 转换为张量
                mtf.ToTensord(keys=["seg"], dtype=torch.int),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
        elif mode == 'test':
            self.transform = val_transform  # 测试模式使用验证变换

        self.cls_questions = PosREC_templates["cls_questions"]  # 分类问题模板
        self.des_qustions = PosREC_templates["des_questions"]  # 描述问题模板
        self.cls_answers = PosREC_templates["cls_answers"]  # 分类答案模板
        self.des_answers = PosREC_templates["des_answers"]  # 描述答案模板
        self.cls_no_answers = PosREC_templates["cls_no_answers"]  # 分类无答案模板
        self.des_no_answers = PosREC_templates["des_no_answers"]  # 描述无答案模板

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]  # 获取数据列表中的数据

            image_path = data['image']  # 获取图像路径
            seg_path = data['label']  # 获取分割路径

            image_array = np.load(image_path)  # 加载图像数据（1*32*256*256, 归一化）
            seg_array = np.load(seg_path)  # 加载分割数据
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])  # 获取分类ID
                # 示例：
                # 假设 seg_path 的值为 /path/to/segmentation_123.npy，那么：
                # (1)os.path.basename(seg_path):
                #    返回 segmentation_123.npy。
                # (2).split('_')[1]:
                #    将 segmentation_123.npy 分割为 ['segmentation', '123.npy']，并获取第二部分 123.npy。
                # (3).split('.')[0]:
                #    将 123.npy 分割为 ['123', 'npy']，并获取第一部分 123。
                # (4)int('123'):
                #    将字符串 123 转换为整数 123。
                # 最终，cls_id 的值为 123。

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)  # 对图像和分割数据应用预定义的变换

                image = it['image']
                seg = it['seg']  # 1*D*H*W（单通道的三维分割数据）

                cls_list = self.dataset_info[self.tag]  # 获取分类列表
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()  # 获取有效分类
                # 示例：
                # 假设 seg 的形状为 (1, 32, 256, 256)。假设 seg 的数据如下：
                #    seg = torch.zeros((1, 32, 256, 256))
                #    seg[0, 5, :, :] = 1  # 第6个通道有分割数据
                #    seg[0, 10, :, :] = 1  # 第11个通道有分割数据
                # (1)计算分割数据在所有维度上的和：torch.sum(seg, dim=(1, 2, 3))
                #    结果是一个形状为 (32,) 的张量：
                #    tensor([0, 0, 0, 0, 0, 65536, 0, 0, 0, 0, 65536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                # (2)获取非零元素的索引：torch.nonzero(torch.sum(seg, dim=(1, 2, 3)))
                #    结果是一个形状为 (N, 1) 的张量，其中 N 是非零元素的数量：
                #    tensor([[5], [10]])
                # (3)展平张量：torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten()
                #    结果是一个形状为 (N,) 的一维张量：
                #    tensor([5, 10])
                # (4)转换为列表：torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                #    结果是一个列表：
                #    [5, 10]
                # 最终，vld_cls 的值是 [5, 10]。
                ### (32,) 表示一个一维张量，包含32个元素，每个元素表示一个通道的总和。
                ### (N, 1) 表示一个二维张量，包含 N 个元素，每个元素是一个长度为1的向量，表示非零元素的索引。
                ### (N,) 表示一个一维张量，包含 N 个元素，表示展平后的非零元素的索引。

                
                if vld_cls:
                    box = mask2box(seg[0])  # 获取分割框
                    if not self.description:  # （分类问题）
                        question_temple = random.choice(self.cls_questions)  # 随机选择一个分类问题模板
                        question = question_temple.format(cls_list[cls_id])  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]  # 构建框文本
                        answer = random.choice(self.cls_answers).format(box_text)  # 构建答案
                    else:  # （描述问题）
                        question_temple = random.choice(self.des_qustions)  # 随机选择一个描述问题模板
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]  # 构建框文本
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], box_text)  # 构建答案
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)  # 随机选择一个分类问题模板
                        question = question_temple.format(cls_list[cls_id])  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])  # 构建无答案
                    else:
                        question_temple = random.choice(self.des_qustions)  # 随机选择一个描述问题模板
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])  # 构建无答案

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )  # 使用分词器对文本进行编码，并将其转换为PyTorch张量

                input_id = text_tensor["input_ids"][0]  # 获取编码后的输入ID
                attention_mask = text_tensor["attention_mask"][0]  # 获取注意力掩码

                valid_len = torch.sum(attention_mask)  # 计算 attention_mask 中所有非零元素的和，即有效标记的数量
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于输入ID长度，则在有效标记位置添加结束标记

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对问题进行编码，并将其转换为PyTorch张量
                question_len = torch.sum(question_tensor["attention_mask"][0])  # 计算问题的有效标记数量

                label = input_id.clone()  # 复制输入ID作为标签
                label[:question_len] = -100  # 将问题部分的标签设置为-100，表示不计算损失
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于标签长度，则在有效标记位置添加结束标记
                else:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100

                ret = {
                    'image': image,  # 图像数据
                    'input_id': input_id,  # 输入ID
                    'label': label,  # 标签
                    'attention_mask': attention_mask,  # 注意力掩码
                    'question': question,  # 问题
                    'answer': answer,  # 答案
                    'question_type': "REC",  # 问题类型
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})  # 如果启用分割，则添加分割数据

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)  # 如果出现异常，则随机选择一个新的索引


class PosREGDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args  # 保存传入的参数
        self.tokenizer = tokenizer  # 分词器

        self.tag = tag  # 数据集标签
        self.mode = mode  # 模式（train/validation/test）
        self.description = description  # 是否使用描述

        self.dataset_info = dataset_info  # 数据集信息

        self.image_tokens = "<im_patch>" * args.proj_out_num  # 图像标记（将 <im_patch> 重复 args.proj_out_num 次）
        self.box_tokens = ["<bx_start>", "<bx_end>"]  # 框标记

        root_path = args.seg_data_path  # 分割数据路径
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )  # 加载训练数据
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )  # 加载验证数据
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )  # 加载测试数据

        # 定义一系列用于数据增强的变换，
        # 这些变换在训练过程中随机应用于图像数据，
        # 以增加数据的多样性，防止模型过拟合，并提高模型的泛化能力。
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),  # 随机旋转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),  # 随机翻转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),  # 随机翻转
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),  # 随机翻转
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),  # 随机缩放强度（缩放因子为0.1,有50%的概率进行缩放）
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),  # 随机偏移强度（偏移量为0.1，有50%的概率进行偏移）
                mtf.ToTensord(keys=["image"], dtype=torch.float),  # 转换为张量
                mtf.ToTensord(keys=["seg"], dtype=torch.int),  # 转换为张量
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),  # 转换为张量
                mtf.ToTensord(keys=["seg"], dtype=torch.int),  # 转换为张量
            ]
        )
        set_track_meta(False)  # 设置不跟踪元数据（减少内存使用并提高处理速度）

        if mode == 'train':
            self.transform = train_transform  # 训练模式使用训练变换
        elif mode == 'validation':
            self.transform = val_transform  # 验证模式使用验证变换
        elif mode == 'test':
            self.transform = val_transform  # 测试模式使用验证变换

        self.cls_questions = PosREG_templates["cls_questions"]  # 分类问题模板
        self.des_questions = PosREG_templates["des_questions"]  # 描述问题模板
        self.cls_answers = PosREG_templates["cls_answers"]  # 分类答案模板
        self.des_answers = PosREG_templates["des_answers"]  # 描述答案模板

        self.cls_no_questions = PosREC_templates["cls_questions"]  # 分类无问题模板
        self.des_no_questions = PosREC_templates["des_questions"]  # 描述无问题模板

        self.cls_no_answers = PosREG_templates["cls_no_answers"]  # 分类无答案模板
        self.des_no_answers = PosREG_templates["des_no_answers"]  # 描述无答案模板

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]  # 获取数据列表中的数据

            image_path = data['image']  # 获取图像路径
            seg_path = data['label']  # 获取分割路径

            image_array = np.load(image_path)  # 加载图像数据（1*32*256*256, 归一化）
            seg_array = np.load(seg_path)  # 加载分割数据
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])  # 获取分类ID

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)  # 对图像和分割数据应用预定义的变换
                image = it['image']
                seg = it['seg']  # 1*D*H*W（单通道的三维分割数据）

                cls_list = self.dataset_info[self.tag]  # 获取分类列表
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()  # 获取有效分类

                if vld_cls:
                    box = mask2box(seg[0])  # 获取分割框
                    if not self.description:  # （分类问题）
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]  # 构建框文本
                        question_temple = random.choice(self.cls_questions)  # 随机选择一个分类问题模板
                        question = question_temple.format(box_text)  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.cls_answers).format(cls_list[cls_id])  # 构建答案
                    else:  # （描述问题）
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]  # 构建框文本
                        question_temple = random.choice(self.des_questions)  # 随机选择一个描述问题模板
                        question = question_temple.format(box_text)  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]]))  # 构建答案
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_no_questions)  # 随机选择一个分类无问题模板
                        question = question_temple.format(cls_list[cls_id])  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])  # 构建无答案
                    else:
                        question_temple = random.choice(self.des_no_questions)  # 随机选择一个描述无问题模板
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))  # 构建问题
                        question = self.image_tokens + ' ' + question  # 构建问题
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])  # 构建无答案

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )  # 使用分词器对文本进行编码，并将其转换为PyTorch张量

                input_id = text_tensor["input_ids"][0]  # 获取编码后的输入ID
                attention_mask = text_tensor["attention_mask"][0]  # 获取注意力掩码

                valid_len = torch.sum(attention_mask)  # 计算 attention_mask 中所有非零元素的和，即有效标记的数量
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于输入ID长度，则在有效标记位置添加结束标记

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )  # 使用分词器对问题进行编码，并将其转换为PyTorch张量
                question_len = torch.sum(question_tensor["attention_mask"][0])  # 计算问题的有效标记数量

                label = input_id.clone()  # 复制输入ID作为标签
                label[:question_len] = -100  # 将问题部分的标签设置为-100，表示不计算损失
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id  # 如果有效标记数量小于标签长度，则在有效标记位置添加结束标记
                else:
                    label[label == self.tokenizer.pad_token_id] = -100  # 将填充标记设置为-100

                ret = {
                    'image': image,  # 图像数据
                    'input_id': input_id,  # 输入ID
                    'label': label,  # 标签
                    'attention_mask': attention_mask,  # 注意力掩码
                    'question': question,  # 问题
                    'answer': answer,  # 答案
                    'question_type': "REG",  # 问题类型
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})  # 如果启用分割，则添加分割数据

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)  # 如果出现异常，则随机选择一个新的索引


class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        # self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiPosDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class PosSegDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosSegDatasets, self).__init__()
        self.ds_list = [
            MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode=mode),
            MultiPosDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



