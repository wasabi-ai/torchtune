# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple
import typing
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import init_process_group
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, utils
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils.activations import apply_selective_activation_checkpointing

from tqdm import tqdm


log = utils.get_logger("DEBUG")


class DistillationRecipeDistributed(FTRecipeInterface):
    """
     distillation recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. DDP is currently not supported. Training on CPU
            is not supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision.  fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def make_checkpointer(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        return checkpointer

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        try:
            if (
                self.seed != ckpt_dict[utils.SEED_KEY]
                or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
                or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
            ):
                warn(
                    message="""Configured value for seed, epochs or max_steps_per_epoch
                    does not match the value stored in checkpoint."""
                )
            self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
            self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
            self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]
        except KeyError as e:
            raise KeyError from e(
                "Checkpoint does not contain the required keys needed for updating recipe state."
                "Are you sure you passed in the right recipe checkpoint?"
            )

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        self._checkpointer = self.make_checkpointer(cfg.student_checkpointer)
        if self._resume_from_checkpoint:
            student_ckpt = self._checkpointer.load_checkpoint()
            self._update_recipe_state(student_ckpt)
        else:
            student_ckpt = None

        teacher_checkpointer = self.make_checkpointer(cfg.teacher_checkpointer)
        teacher_ckpt = teacher_checkpointer.load_checkpoint()

        # ``_setup_models`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the models
        self._teacher_model, self._student_model = self._setup_models(
            teacher_cfg=cfg.teacher_model,
            student_cfg=cfg.student_model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            student_ckpt=student_ckpt[utils.MODEL_KEY] if student_ckpt else None,
            teacher_ckpt=teacher_ckpt[utils.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)

        self.config = cfg
        self._optimizers = []

        def _make_optimizer(layer):
            return self._setup_optimizer(
                cfg_optimizer=cfg.optimizer,
                model=self._student_model,
                parameters=layer.parameters(),
                opt_state_dict=(
                    student_ckpt[utils.OPT_KEY]
                    if self._resume_from_checkpoint
                    else None
                ),
            )

        self._optimizers.append(_make_optimizer(self._student_model.tok_embeddings))
        for i, layer in enumerate(self._student_model.layers):
            self._optimizers.append(_make_optimizer(layer))
        self._optimizers.append(_make_optimizer(self._student_model.output))

        self._combined_optimizer = _make_optimizer(self._student_model)

        self._loss_fn = config.instantiate(cfg.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = len(self._dataloader)
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

    def _setup_models(
        self,
        teacher_cfg: DictConfig,
        student_cfg: DictConfig,
        enable_activation_checkpointing: bool,
        student_ckpt: Dict[str, Any],
        teacher_ckpt: Dict[str, Any],
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> typing.Tuple[nn.Module, nn.Module]:
        """
        Model initialization has some important considerations:
            a. To minimize GPU peak memory, we load the model on CPU with the right
               dtype. To ensure that we don't instantiate ``world_size`` number of models,
               we initialize on meta_device for all ranks other than rank 0.
            b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
               model weights from checkpoint.
            c. While wrapping the model with FSDP, we set ``sync_module_states``
               to TRUE and broadcast module params and buffers from rank 0.
            d. The ``device_id`` param ensures that the FSDP initialization happens on
               the correct device.
        """
        if self._is_rank_zero:
            log.info("FSDP is enabled. Instantiating Model on CPU for Rank 0 ...")
            init_start = time.perf_counter()

            with utils.set_default_dtype(self._dtype):
                teacher_model = config.instantiate(teacher_cfg)
                student_model = config.instantiate(student_cfg)

            log.info(
                f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
            )

            # Load the teacher model weights. This should happen only on Rank 0
            teacher_model.load_state_dict(teacher_ckpt)
            if student_ckpt is not None:
                student_model.load_state_dict(student_ckpt)

        else:
            # For non-zero ranks, load the model on meta device
            with utils.set_default_dtype(self._dtype), torch.device("meta"):
                teacher_model = config.instantiate(teacher_cfg)
                student_model = config.instantiate(student_cfg)

        if self._dtype == torch.bfloat16:
            teacher_model = teacher_model.to(torch.bfloat16)
            student_model = student_model.to(torch.bfloat16)

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        ac_mode = ac_mode
        ac_option = ac_option

        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                student_model,
                ac_mode,
                ac_option,
            )

        # Wrap the model with FSDP. This will ensure that the model is sharded
        # across all available GPUs.
        # def fsdp_wrap(module):
        #     return FSDP(
        #         module=module,
        #         auto_wrap_policy=utils.get_full_finetune_fsdp_wrap_policy(
        #             memory_efficient_fsdp_wrap=memory_efficient_fsdp_wrap,
        #             modules_to_wrap={modules.TransformerDecoderLayer},
        #         ),
        #         sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        #         device_id=self._device,
        #         # this recipe does not currently support mixed precision training
        #         mixed_precision=None,
        #         # Ensure we broadcast params and buffers from rank 0
        #         sync_module_states=True,
        #         # Initialize empty modules on all non-zero ranks
        #         param_init_fn=(
        #             lambda module: (
        #                 module.to_empty(device=torch.device("cuda"), recurse=False)
        #                 if not self._is_rank_zero
        #                 else None
        #             )
        #         ),
        #     )
        # teacher_model = fsdp_wrap(teacher_model)
        # student_model = fsdp_wrap(student_model)
        teacher_model = teacher_model.to(device="cuda")
        student_model = student_model.to(device="cuda")

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(teacher_model)
        utils.validate_no_params_on_meta_device(student_model)

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            utils.set_activation_checkpointing(
                student_model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        if self._is_rank_zero:
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()
        return teacher_model, student_model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        model: nn.Module,
        parameters: typing.Iterable[torch.nn.Parameter],
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:
        """
        Set up the optimizer. This method also handles transforing the state dict
        for FSDP.
        """
        optimizer = config.instantiate(cfg_optimizer, parameters)

        if opt_state_dict:
            opt_state_dict = utils.transform_opt_state_dict(
                opt_state_dict, model, optimizer
            )
            optimizer.load_state_dict(opt_state_dict)

        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._combined_optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._student_model.state_dict().items()}
        ckpt_dict.update({utils.MODEL_KEY: state_dict})
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def train_e2e(self, curr_epoch) -> None:
        print("Training end to end model.")
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._combined_optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        num_tokens = 0

        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        self._sampler.set_epoch(curr_epoch)

        pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
        for idx, batch in enumerate(self._dataloader):
            if (
                self.max_steps_per_epoch is not None
                and (idx) == self.max_steps_per_epoch
            ):
                break

            input_ids, labels = batch
            input_ids = input_ids.to(self._device)
            num_tokens += input_ids.numel()
            labels = labels.to(self._device)

            logits = self._student_model(input_ids)
            # Shift so that tokens < n predict n
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)
            # Compute loss
            loss = self._loss_fn(logits, labels)
            loss.backward()
            self._combined_optimizer.step()
            self._combined_optimizer.zero_grad()

            # Update the number of steps when the weights are updated
            self.global_step += 1

            loss_to_log = loss
            pbar.update(1)
            pbar.set_description(
                f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log}"
            )

            # Log per-step metrics
            if self.global_step % self._log_every_n_steps == 0 and self._is_rank_zero:
                time_per_step = time.perf_counter() - t0
                log_dict = {
                    "loss": loss_to_log,
                    "tokens_per_second": num_tokens / time_per_step,
                }
                if self._log_peak_memory_stats:
                    log_dict.update(utils.get_memory_stats(device=self._device))
                self._metric_logger.log_dict(
                    log_dict,
                    step=self.global_step,
                )

            # Reset running stats for the next step
            num_tokens = 0
            t0 = time.perf_counter()

    def train_layerwise(self, curr_epoch) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        # clean up before training begins
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        for optimizer in self._optimizers:
            optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        self._sampler.set_epoch(curr_epoch)

        pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
        for idx, batch in enumerate(self._dataloader):
            if (
                self.max_steps_per_epoch is not None
                and (idx) == self.max_steps_per_epoch
            ):
                break

            input_ids, labels = batch
            input_ids = input_ids.to(self._device)
            num_tokens += input_ids.numel()
            labels = labels.to(self._device)

            # compute layer-wise updates. for each layer, we compute the teacher activations
            # and then update the student-model to better approximate the parent.
            layer_loss = torch.nn.MSELoss()

            def train_layer(
                inputs, optimizer, teacher_layer, student_layer, loss=layer_loss
            ):
                """Train a single layer. Outputs the loss for the layer and teacher activations."""
                teacher_activations = teacher_layer(inputs).detach()
                student_activations = student_layer(inputs)

                loss = layer_loss(student_activations, teacher_activations)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                return loss, teacher_activations

            loss, activations = train_layer(
                input_ids,
                self._optimizers[0],
                self._teacher_model.tok_embeddings,
                self._student_model.tok_embeddings,
            )
            running_loss += loss

            for layer_idx, student_layer in enumerate(self._student_model.layers):
                loss, activations = train_layer(
                    activations,
                    self._optimizers[layer_idx + 1],
                    self._teacher_model.layers[layer_idx],
                    student_layer,
                )
                running_loss += loss

            activations = self._teacher_model.norm(activations)
            loss, activations = train_layer(
                activations,
                self._optimizers[-1],
                self._teacher_model.output,
                self._student_model.output,
                loss=self._loss_fn,
            )

            # Update the number of steps when the weights are updated
            self.global_step += 1

            loss_to_log = running_loss
            pbar.update(1)
            pbar.set_description(
                f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log}"
            )

            # Log per-step metrics
            if self.global_step % self._log_every_n_steps == 0 and self._is_rank_zero:
                time_per_step = time.perf_counter() - t0
                log_dict = {
                    "loss": loss_to_log,
                    "tokens_per_second": num_tokens / time_per_step,
                }
                if self._log_peak_memory_stats:
                    log_dict.update(utils.get_memory_stats(device=self._device))
                self._metric_logger.log_dict(
                    log_dict,
                    step=self.global_step,
                )

            # Reset running stats for the next step
            running_loss = 0
            num_tokens = 0
            t0 = time.perf_counter()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        torch.distributed.destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed distillation recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )

    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    config.log_config(recipe_name="DistillationRecipeDistributed", cfg=cfg)

    recipe = DistillationRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    for epoch in range(recipe.epochs_run, recipe.total_epochs):
        recipe.train_layerwise(epoch)
        recipe.train_e2e(epoch)
        recipe.epochs_run += 1
        recipe.save_checkpoint(epoch=epoch)

    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
