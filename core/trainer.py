import os
import torch

from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict

from . import constant
from .type_hints import Batch, ConfigDict
from .tools import logger
from .utils import instantiate_from_config, check_dir


class Trainer:
    def __init__(self, model_config: ConfigDict, optimizer_config: ConfigDict, scheduler_config: ConfigDict, loss_group_config: ConfigDict, 
                 sr_scale: int = -1, n_history: int = 0,
                 label: str = "temp", device: Union[str, torch.device] = "auto"):
        super().__init__()
        self._device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        # Config model and optimizer.
        if model_config["params"].get("sr_scale", -1) != sr_scale:
            logger.warning("The generator params \"sr_scale\" is not equal to the given sr_scale.")
        self._sr_scale = sr_scale
        self._n_history = model_config["params"].get("n_history", n_history)
        self._model_location = model_config["target"]

        self._model: nn.Module = instantiate_from_config(model_config)
        self._log_network(self._model)

        optimizer_config["params"]["params"] = self._model.parameters()
        self._optimizer: optim.Optimizer = instantiate_from_config(optimizer_config)

        scheduler_config["params"]["optimizer"] = self._optimizer
        self._scheduler: lr_scheduler.LRScheduler = instantiate_from_config(scheduler_config)
        
        # Config loss function.
        self._loss_group: nn.Module = instantiate_from_config(loss_group_config)
        
        # Check checkpoints and results dir.
        self._ckpt_root = check_dir(os.path.join("checkpoints", label))
        self._results_root = check_dir(os.path.join("results", label))

        # Hyperparameters.
        self._epoch = int(0)
        self._global_step = int(0)
        self._define_runtime_variables()

        self.to(self._device)

    def _define_runtime_variables(self):
        self._lr_frame: Optional[Tensor] = None
        self._hr_frame: Optional[Tensor] = None
        self._aux_inputs: Optional[List[Tensor]] = None

        self._sr_frame: Optional[Tensor] = None
        self._mask_for_loss = None

        self._loss_dict: Dict[str, float] = {}
    
    @property
    def global_step(self):
        return self._global_step

    def compile(self, mode: str = "max-autotune", set_high_precision: bool = True):
        if set_high_precision:
            torch.set_float32_matmul_precision("high")

        logger.info("[Init]", f"Enable model compiled with mode: {mode}.")
        self._model = torch.compile(self._model, mode=mode)

    def train(self, mode: bool = True):
        self._model.train(mode)

    def eval(self):
        self._model.train(False)

    def to(self, device):
        self._model.to(device)
        self._loss_group.to(device)

    def _make_inputs(self, batch: Batch, scale: float) -> Batch:
        for key in list(batch.keys()):
            value = batch[key]
            if isinstance(value, Tensor):
                batch[key] = value.to(self._device)
        
        def _interpolate_sequence(sequence: Tensor, scale: float) -> Tensor:
            b, l, c, th, tw = sequence.shape
            sh, sw = int(th * scale), int(tw * scale)
            return F.interpolate(sequence.view(b * l, c, th, tw), scale_factor=scale, mode="nearest").view(b, l, c, sh, sw)

        self._sr_scale = scale
        inv_scale = 1.0 / scale
        batch["scale"] = scale

        batch["hr_frames"] = batch.get("anti-alias")        
        batch["lr_frames"] = _interpolate_sequence(batch.get("alias"), inv_scale)

        buffers = batch.get("buffers", None)
        if buffers is not None:
            batch["hr_buffers"] = buffers
            batch["lr_buffers"] = _interpolate_sequence(buffers, inv_scale)

        # velocity = batch.get("velocity", None)
        # if velocity is not None:
            # batch["velocity"] = _interpolate_sequence(velocity, inv_scale) * inv_scale
        
        return batch
    
    def _make_one_frame_inputs(self, batch: Batch, frame_idx: int) -> Batch:
        #NOTE: hr_frames: l, lr_frames: h + l, buffers: h + l, velocity: (h - 1) + l
        sub_batch = {"scale": batch["scale"]}

        sub_batch["hr_frames"] = batch["hr_frames"][:, frame_idx]
        sub_batch["lr_frames"] = batch["lr_frames"][:, frame_idx: frame_idx + self._n_history + 1]

        hr_buffers = batch.get("hr_buffers", None)
        if hr_buffers is not None:
            sub_batch["hr_buffers"] = hr_buffers[:, frame_idx: frame_idx + self._n_history + 1]
        
        lr_buffers = batch.get("lr_buffers", None)
        if lr_buffers is not None:
            sub_batch["lr_buffers"] = lr_buffers[:, frame_idx: frame_idx + self._n_history + 1]

        velocity = batch.get("velocity", None)
        if velocity is not None:
            sub_batch["velocity"] = velocity[:, frame_idx: frame_idx + self._n_history]

        sr_frames = batch.get("sr_frames", None)
        if sr_frames is not None and len(sr_frames) > 0:
            if self._n_history == 1:
                sub_batch["sr_frames"] = sr_frames[-1]
            else:
                pass #TODO: sr_frames
  
        return sub_batch

    def training_step(self, batch: Batch, batch_idx: int, scale: float):
        batch = self._make_inputs(batch, scale)
        batch["sr_frames"] = []

        self._loss_dict.clear()

        sequence_length = batch["hr_frames"].shape[1]
        for idx in range(sequence_length):
            one_frame_batch = self._make_one_frame_inputs(batch, idx)
            one_frame_batch = self._model(one_frame_batch)

            if constant.ENABLE_PIXEL_CLIP:
                one_frame_batch["sr_frames"] = torch.clip(
                    one_frame_batch["sr_frames"], min=constant.CLIP_MIN_PIXEL, max=constant.CLIP_MAX_PIXEL)

            self._optimizer.zero_grad()
            loss, loss_dict = self._loss_group(one_frame_batch)
            loss.backward()
            self._optimizer.step()

            batch["sr_frames"].append(one_frame_batch["sr_frames"].detach())
            for key, value in loss_dict.items():
                if key not in self._loss_dict:
                    self._loss_dict[key] = []
                self._loss_dict[key].append(value)

        for key, value in self._loss_dict.items():
            self._loss_dict[key] = sum(value) / len(value)
  
        self._global_step += 1
        if self._global_step % constant.LOG_LOSS_INTERVAL == 0:
            self._log_loss_dict(batch_idx)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def update_lr(self):
        curr_lr = self._optimizer.state_dict()['param_groups'][0]['lr']
        self._scheduler.step()
        next_lr = self._optimizer.state_dict()['param_groups'][0]['lr']

        if curr_lr != next_lr:
            logger.info(f"Update lr: {curr_lr:.9f} -> {next_lr:.9f}")
    
    def _log_network(self, network: nn.Module):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        network_desc = str(network)
        num_params = sum(map(lambda x: x.numel(), network.parameters()))

        logger.info("[Init]", f"Create network: {network._get_name()} ({self._model_location}), parameters: {num_params}")
        logger.record("[Init]", network_desc)

    def _log_loss_dict(self, batch_idx: int):
        loss_msg = f"epoch: {self._epoch}, step: {batch_idx + 1}, global_step: {self._global_step}, x{self._sr_scale}"
        for key, value in self._loss_dict.items():
            loss_msg += f", {key}: {value:.4f}"
        logger.record(loss_msg)

    def print_loss(self, batch_idx: int, batches_per_epoch: int):
        loss_msg = f"epoch: {self._epoch}, step: {batch_idx + 1}/{batches_per_epoch}, x{self._sr_scale}"
        for key, value in self._loss_dict.items():
            loss_msg += f", {key}: {value:.4f}"
        print(loss_msg)
    
    @torch.no_grad()
    def visualize_curr_bacth(self, batch: Batch, scale: float, dirname: str = "train"):
        dir_path = check_dir(os.path.join(self._results_root, dirname))
        file_path = os.path.join(dir_path, f"e-{self._epoch}_s-{self._global_step}_x{scale}.png")

        sr_frames = batch["sr_frames"]
        sr_frames = torch.stack(batch["sr_frames"], dim=1).view(-1, *sr_frames[0].shape[1:])
        hr_frames = batch["hr_frames"].view(-1, *batch["hr_frames"].shape[2:])

        frames = torch.concat((sr_frames, hr_frames), dim=-2)
        frames = torch.pow(frames, 1.0 / constant.GAMMA_CORRECTION)
        save_image(frames, file_path, padding=2, normalize=False)

    def save_state_dict(self, hyperparameters: Optional[Dict[str, Any]] = None, last: bool = False):
        if isinstance(self._model, nn.DataParallel) or isinstance(self._model, nn.parallel.DistributedDataParallel):
            self._model = self._model.module

        if last:
            ckpt_name = f"{self._model._get_name()}_{self._sr_scale}x_last.ckpt"
        else:
            ckpt_name = f"{self._model._get_name()}_{self._sr_scale}x_epoch-{self._epoch}_steps-{self.global_step}.ckpt"
        ckpt_path = os.path.join(self._ckpt_root, ckpt_name)

        state_dict = self._model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        ckpt = {"state_dict": state_dict}
        if hyperparameters is not None:
            ckpt.update(hyperparameters)
        torch.save(ckpt, ckpt_path)

    def load_state_dict(self, ckpt: Dict[str, Any], strict: bool = True, ignore_prefix: str = "module._orig_mod."):
        if isinstance(self._model, nn.DataParallel) or isinstance(self._model, DistributedDataParallel):
            self._model = self._model.module

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
            
        ckpt_clean = OrderedDict()  # remove unnecessary 'module.'
        for key, param in ckpt.items():
            if key.startswith(ignore_prefix):
                key = key[len(ignore_prefix):]
            ckpt_clean[key] = param

        missing_keys, unexpected_keys = self._model.load_state_dict(ckpt_clean, strict=strict)
        print("[Init] Successfully loaded state dict.")
        if missing_keys:
            logger.warning(f"[Init] Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"[Init] Unexpected keys: {unexpected_keys}")

    def save_checkpoint(self):
        hyperparameters = {
            "epoch": self._epoch,
            "global_step": self._global_step,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict()
        }
        self.save_state_dict(hyperparameters, last=True)

    def load_checkpoint(self, ckpt_path: str, resume_epoch: int, strict: bool = True):
        ckpt = torch.load(ckpt_path)
        assert int(ckpt["epoch"] + 1) == resume_epoch, \
            f"resume epoch is not equal to checkpoint epoch. resume_epoch: {resume_epoch}, ckpt_epoch: {ckpt['epoch'] + 1}"

        self._epoch = resume_epoch
        self._global_step = ckpt["global_step"]
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._scheduler.load_state_dict(ckpt["scheduler"])
        self.load_state_dict(ckpt, strict)
        
        lr = self._optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f"[Init] Resume checkpoint from {ckpt_path}. Epoch: {self._epoch}, global_step: {self._global_step}, learning rate: {lr:.7f}")
