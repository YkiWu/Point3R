# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# CUT3R: https://github.com/CUT3R
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True 

from dust3r.point3r import (
    PreTrainedModel,
    Point3R,
    Point3RConfig,
    inf,
    strip_module,
    from_dust3r_to_ours,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now() 
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )

    # model
    printer.info("Loading model: %s", args.model)
    model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(
        f"Encoder parameters: {sum(p.numel() for p in model.enc_blocks.parameters())}"
    )
    printer.info(
        f"Decoder parameters: {sum(p.numel() for p in model.dec_blocks.parameters())}"
    )

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    
    model.to(device)

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        printer.info(
            model.load_state_dict(from_dust3r_to_ours(strip_module(ckpt["model"])), strict=False)
        )
        del ckpt 
    
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def write_log_stats(epoch, train_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # no test
        new_best = False
        
        # log 
        write_log_stats(epoch, train_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.epochs:
            break  
        
        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
                point3r_tag=True,
            )
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            del loss
            del batch

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)

            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  

                if log_writer is None:
                    continue
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"))

    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
