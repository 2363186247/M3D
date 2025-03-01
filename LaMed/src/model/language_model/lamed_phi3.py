from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM


# 继承 Phi3Config 并添加一个新的 model_type 属性，为 LamedPhi3 模型提供了一个特定的配置类。
# 使得在使用 AutoConfig 和 AutoModelForCausalLM 时，可以正确识别和加载 LamedPhi3 模型的配置。
class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"


# 
class LamedPhi3Model(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config  
    def __init__(self, config: Phi3Config):
        super(LamedPhi3Model, self).__init__(config)  # 调用父类的初始化函数


class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLM, self).__init__(config)
        self.model = LamedPhi3Model(config)     # 创建 LamedPhi3Model 模型
        self.vocab_size = config.vocab_size     # 词表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 线性层

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        # 尝试获取非零的分割 ID，如果失败则设置为空列表。
        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []

        if self.get_model().seg_enable and seg_ids:     # 如果分割模块启用且存在非零分割 ID
            # 调用父类的 forward 方法，获取输出
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,

                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states    # 获取隐藏状态

            last_hidden_state = output_hidden_states[-1]    # 获取最后一个隐藏状态

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
                # 与 self.config.seg_token_id 进行比较，生成一个布尔掩码，表示哪些位置是分割 token。
                # input_ids_pre 是输入的 token ID 序列，形状为 (batch_size, sequence_length)。
                # 通过 [:, 1:]，获取从第二个 token 开始的子序列，形状为 (batch_size, sequence_length - 1)。
                # 去掉第一个 token 是为了更准确地处理分割 token 掩码。
                # 避免特殊 token 的干扰，并确保模型在处理分割 token 时的一致性。

            # 通过扩展掩码的维度，确保掩码的形状与隐藏状态的形状一致
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                        # torch.zeros：创建一个全零的张量。
                        # (seg_token_mask.shape[0], 1)：指定张量的形状，行数与 seg_token_mask 的行数相同，列数为 1。
                        # dtype=seg_token_mask.dtype：指定张量的数据类型与 seg_token_mask 相同
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:   # 单个分割 token
                    seg_token = last_hidden_state[i][seg_token_mask[i]] # 通过掩码选择最后一个隐藏状态中对应分割 token 的隐藏状态。
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:  # 多个分割 token
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]    
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True) # 对多个分割 token 的隐藏状态求平均
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:   # 无分割 token
                    # 创建一个全零张量，数据类型与 last_hidden_state 相同
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)
                # 示例：
                # last_hidden_state = torch.tensor([
                #     [[0.1, 0.2, ..., 0.768], [0.3, 0.4, ..., 0.768], [0.5, 0.6, ..., 0.768], [0.7, 0.8, ..., 0.768], [0.9, 1.0, ..., 0.768]],
                #     [[1.1, 1.2, ..., 0.768], [1.3, 1.4, ..., 0.768], [1.5, 1.6, ..., 0.768], [1.7, 1.8, ..., 0.768], [1.9, 2.0, ..., 0.768]]
                # ])
                # seg_token_mask = torch.tensor([
                #     [False, True, False, False, False],
                #     [False, False, True, False, False]
                # ])
                # last_hidden_state[0] 表示第一个样本的隐藏状态，形状为 (5, 768)。
                # seg_token_mask[0] 表示第一个样本的分割 token 掩码，形状为 (5,)，其中第二个位置为 True，表示这是一个分割 token。

            seg_prompts = torch.cat(seg_prompts, dim=0) # 将所有分割 token 的提示张量拼接在一起
            logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
            loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])   #  Dice 损失
            loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])     # 二元交叉熵损失
            seg_loss = loss_dice + loss_bce  # 总损失
            outputs.loss = outputs.loss + seg_loss
            return outputs
        else:   # 如果分割模块未启用或不存在非零分割 ID
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:  # 返回值类型为 GenerateOutput、torch.LongTensor 或 Any
        position_ids = kwargs.pop("position_ids", None) # 移除并返回字典中指定键的值
        attention_mask = kwargs.pop("attention_mask", None)
        
        # 确保 inputs_embeds 参数不会被传递给方法。
        # 如果用户尝试传递 inputs_embeds 参数，方法会立即抛出异常并停止执行。
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,      # 返回隐藏状态
                return_dict_in_generate=True,   # 在生成时返回字典
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id  # 获取分割 token 掩码

            last_tensors = [tuple[-1] for tuple in output_hidden_states]    # 提取最后一个隐藏状态
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)          # 沿维度 1 进行拼接

            # 根据 seg_token_mask 生成分割 token 的提示张量 seg_prompts。
            # 对每个样本，如果只有一个分割 token，则直接使用该分割 token 的隐藏状态生成提示张量；
            # 如果有多个分割 token，则对这些分割 token 的隐藏状态求平均后生成提示张量；
            # 如果没有分割 token，则生成一个全零的提示张量，并记录该样本的索引。
            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:   # 单个分割 token
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:  # 多个分割 token
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:   # 无分割 token
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0) # 沿维度 0 拼接
            logits = self.get_model().seg_module(images, seg_prompts)   #计算分割模块的输出 logits
            logits[noseg_ids] = -torch.inf  # 无分割 token 的 logits 设置为负无穷

            return output_ids, logits
        else:   # 如果分割模块未启用
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return output_ids

    # 重写 prepare_inputs_for_generation 方法，添加 images 参数。
    # 该方法用于准备生成文本的输入，将 images 参数添加到输入字典中。
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_phi3", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)