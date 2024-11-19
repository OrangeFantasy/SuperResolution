from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypedDict, Union
from numpy import ndarray
from torch import Tensor


@dataclass(frozen=True)
class FrameIndexInfo:
    idx: int
    idx_str: str
    str_width: int
    full_name: str

@dataclass(frozen=True)
class ImageSize:
    w: int
    h: int

@dataclass(frozen=True)
class CropRect:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass(frozen=True)
class TransformParams:
    hflip: bool
    vflip: bool
    rot90: bool


TransformFunc = Callable[[ndarray, TransformParams], ndarray]


class ConfigDict(TypedDict):
    target: str
    params: Dict[str, Any]


Batch = Dict[str, Union[Tensor, List[List[str]]]]




