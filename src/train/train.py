# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib

import torch

import transformers

from src.utils.dataset import make_supervised_data_module
from src.utils.arguments import *
import src.utils.conversation as conversation_lib

from src.model.aligngpt import AlignGPTForCausalLM
from src.model.aligngpt_mistral import AlignGPTMistralForCausalLM
from src.model.aligngpt_phi import AlignGPTPhiForCausalLM
from src.model.aligngpt_qwen2 import AlignGPTQwen2ForCausalLM
from src.model.aligngpt_gemma import AlignGPTGemmaForCausalLM

from src.train.aligngpt_trainer import AlignGPTTrainer
from src.utils.constants import *

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# =====================================================================================================
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


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # 保存 adapter 和 align 相关参数
        keys_to_match = ["mm_projector", 
                         'indicator_embs', 
                         ]
        if getattr(trainer.args, "use_im_start_end", False):
            # TODO：不进入
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), 
                                                           keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                # TODO：不进入
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                # bin_file_name = f'mm_projector.bin'
                bin_file_name = f'mm_projector_align.bin'
                torch.save(weight_to_save, os.path.join(output_dir, bin_file_name))
        return

    # =====================================================================
    # 微调时从这里进行存储
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

# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    # =====================================================================
    # 设置 device_map
    # zero3 时不能使用 device map
    # TODO：如果不用 device map，注释掉下面这块代码
    if 'zero3' not in training_args.deepspeed:
        # print('Not using ZeRO3') # ？？？运行时需要注释掉，打印信息会影响 learning rate
        
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device}
        ))
    
    # =====================================================================
    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # =====================================================================
    # 根据 version 设置 padding token，并且选择用于处理对话的模板
    if 'llama-3' in model_args.model_name_or_path:
        print('here set llama_3')
        # 新加的
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token
        
        # tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
    
    # 加载多模态模型
    # 实质上只加载了预训练的 backbone LLM
    if model_args.vision_tower is not None:
        if "phi-1_5" in model_args.model_name_or_path.lower() or "phi-2" in model_args.model_name_or_path.lower():
            model = AlignGPTPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
        elif "vicuna" in model_args.model_name_or_path.lower():
            model = AlignGPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "mistral" in model_args.model_name_or_path.lower():
            model = AlignGPTMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            model = AlignGPTQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = AlignGPTGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "llama-3-8b" in model_args.model_name_or_path.lower():
            model = AlignGPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
    else:
        raise ValueError(f'Shouldn\'t reach here')
    model.config.use_cache = False
    model.config.n_indicators = training_args.n_indicators # assign the number of indicators

    # =====================================================================
    # 冻结 backbone LLM 的参数
    # TODO：不会用到，因为后面会冻结整个模型的参数
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # =====================================================================
    # 进行 gradient checkpoint 设置
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if model_args.version in conversation_lib.conv_templates:
        # 预训练的是 plain --> conv_llava_plain
        # 微调是 v1 --> conv_vicuna_v1
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # =====================================================================
    # 设置模型处于的训练阶段：pretrain、finetune、inference
    model.set_stage(training_args.stage)
    
    # =====================================================================
    # 初始化 or 进一步加载
    # 视觉模块、多模态对齐模块的参数
    if model_args.vision_tower is not None:
        model.initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        model.initialize_align_components() # 初始化对齐度模块
        # print(model.align_indicators.indicator_embs.shape)
        model.load_pretrained_weights(model_args=model_args) # 加载预训练阶段的参数
        
        vision_tower = model.get_vision_tower() # vision tower 的参数在创建时冻结
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        # ===============================================
        # 用于预训练阶段
        # 在这里，冻结整个模型参数更新
        # 然后启用 mm_projector 以及对齐度相关模块的参数更新
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False) # 冻结了整个模型所有参数
            for p in model.mm_projector.parameters(): # 把 mm_projector 层的启用更新
                p.requires_grad = True
            
            model.align_indicators.indicator_embs.requires_grad = True
             
        # ===============================================
        # 冻结 mm_projector 的参数
        # TODO：没有用到这个参数
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        
        # ===============================================
        # 在 finetune 阶段，
        # 只需要冻结 align indicators 的参数更新
        # if training_args.stage == FINETUNE:
        if 'finetune' in training_args.stage:
            model.align_indicators.indicator_embs.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # =====================================================================
    # 加载并且根据 version 等参数来处理数据
    rank0_print("Formatting inputs...Skip in lazy mode")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # =====================================================================
    # 定义训练器，然后训练
    trainer = AlignGPTTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # =====================================================================
    # 存储参数
    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()