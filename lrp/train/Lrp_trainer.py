
import os
import torch
import torch.nn as nn
import json
from torch.utils.data import Sampler
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from torch.utils.data import DataLoader, Subset
import random
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from lrp.model.utils import LrpConfig, _get_submodules, LlmLrpTSLinear, VitLrpTSLayerNorm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp




def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        if hasattr(param, 'ds_active_sub_modules') and param.ds_active_sub_modules:
            param.ds_active_sub_modules.clear()
        with zero.GatheredParameters([param], enabled=True):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    grad = {k: t.requires_grad for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    new_dic = {}
    for key in to_return.keys():
        if 'weight' in key or 'bias' in key:
            new_dic[key] = to_return[key]
    return new_dic

def get_lrp_state_maybe_zero_3(named_params, keys_to_match, if_merge = False):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    grad = {k: t.requires_grad for k, t in to_return.items()}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    if not if_merge:
        return to_return
    merge_pairs = []
    processed_keys = set()
    for key in to_return.keys():
        if key in processed_keys:
            continue
        if 'static_' in key:
            corresponding_train_key = key.replace('static_', 'train_')
            merge_pairs.append((key, corresponding_train_key))
            processed_keys.add(key)
            processed_keys.add(corresponding_train_key)
        elif 'train_' in key:
            corresponding_static_key = key.replace('train_', 'static_')
            merge_pairs.append((corresponding_static_key, key))
            processed_keys.add(key)
            processed_keys.add(corresponding_static_key)
    merged_dict = {}
    for static_key, train_key in merge_pairs:
        static_tensor = to_return.get(static_key)
        train_tensor = to_return.get(train_key)
        if static_tensor is not None and train_tensor is not None:
            if 'rank_A' in static_key:
                merged_tensor = torch.cat([static_tensor, train_tensor], dim=1)
            else:
                merged_tensor = torch.cat([static_tensor, train_tensor], dim=0)
            merged_dict[static_key] = merged_tensor
        elif static_tensor is not None:
            merged_dict[static_key] = static_tensor
        elif train_tensor is not None:
            merged_dict[static_key] = train_tensor
    unmerged_keys = set(to_return.keys()) - processed_keys
    for key in unmerged_keys:
        merged_dict[key] = to_return[key]
    save_dict = {}
    for key in merged_dict.keys():
        if "teacher_" not in key:
            save_dict[key] = merged_dict[key]
    return save_dict
    

def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]



class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
    def __len__(self):
        return len(self.lengths)
    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)




class LrpTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, 
                lengths=lengths, 
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None or self.args.prompt_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                prompt_parameters = [name for name, _ in opt_model.named_parameters() if "prompt" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in prompt_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in prompt_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in prompt_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.prompt_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in prompt_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.prompt_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        mm_keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(self.args, "use_im_start_end", False):
            mm_keys_to_match.extend(['embed_tokens', 'embed_in'])
        mm_weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), mm_keys_to_match)
        lrp_keys_to_match = ['task_llm_query_pool', 'task_vit_query_pool', 'static_keys_llm', 'static_keys_vit', 'train_keys_llm', 'train_keys_vit', 'static_prompt_pool', 'train_prompt_pool', 
                             'share_rank_A_pool', 'share_rank_B_pool', 'static_rank_A_pool', 'static_rank_B_pool', 'train_rank_A_pool', 'train_rank_B_pool']
        lrp_weight_to_save = get_lrp_state_maybe_zero_3(self.model.named_parameters(), lrp_keys_to_match, if_merge = False)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            self.model.lrp_config.save_pretrained(output_dir)
            torch.save(mm_weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            torch.save(lrp_weight_to_save, os.path.join(output_dir, f'lrp.bin'))

            
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LrpTrainer, self)._save(output_dir, state_dict)
    
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            if self.model_accepts_loss_kwargs:
                loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            loss_res = 0
            if self.model.lrp_config.loss_weight1 is not None and self.model.lrp_config.loss_weight1 > 0:
                loss1 = self.model.get_k_loss()
                loss_res += self.model.lrp_config.loss_weight1 * loss1
            if self.model.lrp_config.loss_weight2 is not None and self.model.lrp_config.loss_weight2 > 0:
                loss2 = self.model.get_v_loss()
                loss_res += self.model.lrp_config.loss_weight2 * loss2
            if self.model.lrp_config.loss_weight3 is not None and self.model.lrp_config.loss_weight3 > 0:
                loss3 = self.model.get_q_loss()
                print(loss3)
                loss_res += self.model.lrp_config.loss_weight3 * loss3
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if loss_res > 0:
            self.accelerator.backward(loss_res, **kwargs)
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss