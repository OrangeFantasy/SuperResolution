import importlib
import os
import json
from omegaconf import OmegaConf
from typing import Any, Dict

import torch
from torch.nn import functional as F

from .type_hints import ConfigDict


def sys_setting(seed: int, precision: str = "highest", enable_cudnn: bool = False):
    os.environ['PYTHONHASHSEED'] = str(seed)
                                       
    import random
    random.seed(seed)
    
    import numpy
    numpy.random.seed(seed)

    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn
    cudnn.deterministic = not enable_cudnn
    cudnn.benchmark = enable_cudnn
    cudnn.enabled = enable_cudnn


def load_yaml_and_convert(path: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path))


def instantiate_from_config(config: ConfigDict):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    
    target = config["target"]
    params = config.get("params", dict())

    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**params)


def parse_params(args_str: str) -> Dict[str, Any]:
    return json.loads(args_str)


def check_dir(path: str, path_is_file: bool = False) -> str:
    assert path is not None, "Check the path to make sure it is not none."

    dirname = path
    if path_is_file:
        dirname = os.path.dirname(path)

    os.makedirs(dirname, exist_ok=True)
    return dirname


def check_default(input, default):
    return input if input is not None else default


def upscale_by_shuffle(image: torch.Tensor, scale: int) -> torch.Tensor:
    assert image.dim() == 4, "The dimension of image must be 4."
    assert image.shape[1] == 3, "The number of channels of image must be 3."
    
    r = torch.stack([image[:, 0] for _ in range(scale**2)], dim=1)
    g = torch.stack([image[:, 1] for _ in range(scale**2)], dim=1)
    b = torch.stack([image[:, 2] for _ in range(scale**2)], dim=1)
    
    image = torch.concat([r, g, b], dim=1)
    return F.pixel_shuffle(image, scale)