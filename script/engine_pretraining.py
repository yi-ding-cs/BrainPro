# --------------------------------------------------------
# Towards Large-Scale Cognition-aware EEG Representation Learning
# By Ding Yi
# Based on LaBraM, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM/tree/main
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import pickle
from timm.models import create_model
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from pretraining_data import PreprocessedEEGDataset
import utils
import models.BrainPro.BrainPro
from typing import Iterable
import torch
from einops import rearrange
from contextlib import nullcontext
from models.BrainPro.BrainPro import loss_fun_enhanced


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False
    )
    return model


def train(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)

    dataset = PreprocessedEEGDataset(args.merged_pretraining_data_path)

    if args.distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)
        global_rank = None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        multiprocessing_context='spawn',
    )

    num_training_steps_per_epoch = len(data_loader_train)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # load meta data for useful information
    with open(args.meta_data_path, "rb") as f:
        metadata = pickle.load(f)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            meta_infor=metadata,
            args=args,
        )
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # mask = np.hstack([
    #     np.zeros(len_keep),
    #     np.ones(L - len_keep),
    # ])
    # np.random.shuffle(mask)

    return mask.to(torch.bool)


def group_and_trim_channels(samples: torch.Tensor, labels: torch.Tensor):
    """
    Groups samples by label and removes zero-padded channels across the C dimension.
    Assumes all samples with the same label share the same channel padding pattern
    (i.e., padding is consistent within each label group).

    Args:
        samples: Tensor of shape (B, C, T+2)
        labels: Tensor of shape (B, 2), where labels[:, 0] = dataset_ID

    Returns:
        dict[int, Tensor]: label -> (N_label, C_valid, T+2)
    """

    dataset_ID = labels[:, 0]
    unique_labels = torch.unique(dataset_ID)
    grouped = {}

    for each_dataset_ID in unique_labels:
        each_dataset_ID = int(each_dataset_ID.item())
        mask = dataset_ID == each_dataset_ID  # shape: (B,)
        selected_samples = samples[mask]  # shape: (N_each_dataset, C, T)
        selected_labels = labels[mask, -1]  # shape: (N_each_dataset,)  Select the group index as the y_cognition

        # Use first sample to determine valid channels
        valid_channels = selected_samples[0, :, -2] != 0  # shape: (C,)
        trimmed_samples = selected_samples[:, valid_channels, :]  # shape: (N_label, C_valid, T)

        grouped[each_dataset_ID] = (trimmed_samples, selected_labels)

    return grouped


def get_channel_importance(meta_infor, y_cognition):
    if y_cognition == 0:
        channel_importance = meta_infor.get('affect_important_channel')
    elif y_cognition == 1:
        channel_importance = meta_infor.get('motor_important_channel')
    else:
        channel_importance = torch.ones(len(meta_infor.get('predefined_channels')))
    return channel_importance


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, meta_infor=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (batch) in enumerate(
            metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, labels_ID = batch

        # Group samples by label and trim padded channels
        grouped_samples = group_and_trim_channels(samples, labels_ID)

        total_loss = 0
        total_loss_rec = 0
        total_loss_decouple = 0
        num_groups = len(grouped_samples)

        for i, (idx_dataset, sub_batch) in enumerate(grouped_samples.items()):
            eeg, y_cognition = sub_batch
            if eeg.numel() == 0:
                continue  # skip empty groups
            eeg = eeg.float().to(device, non_blocking=True)
            y_cognition = y_cognition[0]

            channel_importance = get_channel_importance(meta_infor, y_cognition)
            channel_importance = torch.tensor(channel_importance).long().to(device, non_blocking=True)
            y_cognition = torch.tensor(y_cognition).long().to(device, non_blocking=True)

            sync_context = model.no_sync() if args.distributed and i < num_groups - 1 else nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast():
                    channel_idx = eeg[0, :, -2].long()
                    area_idx = eeg[0, :, -1].long()
                    eeg = eeg[:, :, :-2] / 100
                    bool_masked_pos = random_masking(rearrange(eeg, 'B C (N L) -> B (C N) L', L=args.patch_size),
                                                     mask_ratio=0.5).to(device, non_blocking=True)
                    outputs = model(eeg, channel_idx, area_idx, y_cognition, bool_masked_pos, args.patch_size)
                    z_shared, z_cognition, z_other_types, z_target, z_rec = outputs

                    channel_importance = channel_importance[channel_idx-1]
                    loss_group, loss_rec, loss_decouple = loss_fun_enhanced(
                        channel_importance=channel_importance,
                        z_shared=z_shared, z_cognition=z_cognition,
                        z_other_types=z_other_types,
                        z_target=z_target, z_rec=z_rec,
                        T=epoch, Lambda1=1.0, Lambda2=1.0)

                loss_group = loss_group / (args.gradient_accumulation_steps * num_groups)
                total_loss_rec += loss_rec.item() / num_groups
                total_loss_decouple += loss_decouple.item() / num_groups
                total_loss += loss_group.item() * args.gradient_accumulation_steps

                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss_group, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=((step + 1) % args.gradient_accumulation_steps == 0 and i == num_groups - 1))

        loss_value = total_loss
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_rec=total_loss_rec)
        metric_logger.update(loss_decouple=total_loss_decouple)

        if log_writer is not None:
            log_writer.update(loss_rec=total_loss_rec, head="loss")
            log_writer.update(loss_decouple=total_loss_decouple, head="loss")

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
