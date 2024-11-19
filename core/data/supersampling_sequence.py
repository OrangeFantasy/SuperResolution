import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

from .. import constant
from ..type_hints import FrameIndexInfo, ImageSize, CropRect, TransformParams, TransformFunc
from ..tools import logger


def check_nparray(path: str) -> str:
    assert os.path.exists(path) and os.path.splitext(path)[-1].lower() in [".npy", ".npz"], \
        "The file does not exist or is not a valid image array: %s" % path
    return path


class SuperSamplingDataset_Sequence(Dataset):
    # NOTE: Sequence format.
    # sequence       : ..., t_-2, t_-1,  t_0,  t_1,  t_2, ...
    # anti-alias (gt): ..., null, null,    √,    √,    √, ...
    # alias (input)  : ...,    √,    √,    √,    √,    √, ...
    # buffers        : ...,    √,    √,    √,    √,    √, ...
    # velocity       : ...,  null,   √,    √,    √,    √, ...
    # basenames      : ...,  null, null,   √,    √,    √, ...
    
    def __init__(self, anti_alias_root: str, alias_root: str, buffer_root: str, velocity_root: str,
                 n_history_frames: int = 0, max_sequence_length: int = 1, 
                 n_frames: Optional[int] = None, frame_range: Optional[List[Tuple[int, int]]] = None, 
                 image_size: Optional[ImageSize] = (1080, 1920), crop_size: Optional[ImageSize] = (360, 640), 
                 transform_ratio: float = 0.5, phase: str = "train", scene: str = "Unknown") -> None:
        super().__init__()
        self._anti_alias_root = anti_alias_root
        self._alias_root = alias_root
        self._buffer_root = buffer_root
        self._velocity_root = velocity_root

        self._n_history_frames = n_history_frames
        self._max_sequence_length = max_sequence_length

        self._image_size = image_size
        self._crop_size = crop_size

        self._transform_ratio = transform_ratio
        self._phase = phase
        self._scene = scene

        data_table = self._build_data_table()
        if n_frames is None and frame_range is None:
            self._data_table = data_table
        else:
            if frame_range is not None:  # data_range: [[s1, e1], [s2, e2], ...]
                self._data_table = []
                
                for start, end in frame_range:
                    for i in range(start, end):
                        self._data_table.append(data_table[i])
                        if n_frames != -1 and len(self._data_table) >= n_frames:
                            break
                    else:
                        continue
                    break
            else:
                self._data_table = data_table[:n_frames]

        self._len = len(self._data_table)
        logger.info("[Data]", f"Create {phase} dataset. Dataset size: {self._len}")

    @staticmethod
    def _get_frame_index_info(frame_name: str) -> FrameIndexInfo:
        # Image nale format: ScaneName.FrameIndex.RenderPass.Extension. Example: RedwoodForeast.0020.FinalImage.exr.
        frame_idx_str = frame_name.split(".")[-3]
        frame_idx_str_width = len(frame_idx_str)
        frame_idx = int(frame_idx_str)
        return FrameIndexInfo(frame_idx, frame_idx_str, frame_idx_str_width, frame_name)

    def _has_enough_history_frames(self, idx: FrameIndexInfo) -> bool:
        # frame_idx, frame_idx_str, frame_idx_str_width = self._get_frame_index(frame_name)
        for i in range(1, max(self._n_history_frames, constant.MAX_HISTORY_FRAMES) + 1):
            path = os.path.join(
                self._alias_root, idx.full_name.replace(idx.idx_str, str(idx.idx - i).zfill(idx.str_width)))
            if not os.path.exists(path):
                return False
        return True
        
    def _get_sequence_index(self, idx: FrameIndexInfo) -> int:
        # frame_idx, frame_idx_str, frame_idx_str_width = self._get_frame_index(frame_name)
        min_idx = idx.idx - self._n_history_frames
        max_idx = idx.idx + self._max_sequence_length - 1

        for i in range(1, self._max_sequence_length):
            path = os.path.join(
                self._alias_root, idx.full_name.replace(idx.idx_str, str(idx.idx + i).zfill(idx.str_width)))
            if not os.path.exists(path):
                max_idx = idx.idx + (i - 1)
                break
        
        return min_idx, max_idx
      
    def _build_data_table(self) -> List[Dict[str, List[str]]]:
        data_table = []
        
        for start_name in sorted(os.listdir(self._alias_root)):
            idx_info = self._get_frame_index_info(start_name)
            if not self._has_enough_history_frames(idx_info):
                continue

            min_idx, max_idx = self._get_sequence_index(idx_info)
            
            data_row = { "anti-alias": [], "alias": [], "buffers": [], "velocity": [], "basenames": [] }
            for frame_idx in range(min_idx, max_idx + 1):
                idx_str = str(frame_idx).zfill(idx_info.str_width)
                frame_name = start_name.replace(idx_info.idx_str, idx_str)
                
                data_row["alias"].append(check_nparray(os.path.join(self._alias_root, frame_name)))
                data_row["buffers"].append(check_nparray(os.path.join(self._buffer_root, frame_name)))

                if frame_idx >= idx_info.idx:
                    data_row["anti-alias"].append(check_nparray(os.path.join(self._anti_alias_root, frame_name)))
                    data_row["basenames"].append(os.path.splitext(start_name)[0])

                if frame_idx != min_idx:
                    data_row["velocity"].append(check_nparray(os.path.join(self._velocity_root, frame_name)))

            data_table.append(data_row)
        return data_table

    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, List[str]]]:
        data_row = self._data_table[index]

        if self._phase == "train":
            x = random.randint(0, self._image_size.w - self._crop_size.w)
            y = random.randint(0, self._image_size.h - self._crop_size.h)
            crop_rect = CropRect(x, y, x + self._crop_size.w, y + self._crop_size.h)
            transform = self._get_random_transform_params(rotation=False, ratio=self._transform_ratio)
        else:
            crop_rect, transform = None, None

        return_data = {
            "anti-alias": self._load_and_augment(data_row["anti-alias"], crop_rect, transform, self._fn_transform), 
            "alias": self._load_and_augment(data_row["alias"], crop_rect, transform, self._fn_transform), 
            "buffers": self._load_and_augment(data_row["buffers"], crop_rect, transform, self._fn_transform),
            "velocity": self._load_and_augment(data_row["velocity"], crop_rect, transform, self._fn_transform_velocity),
            "basenames": data_row["basenames"]
        }
        return return_data
    
    @staticmethod
    def _load_and_augment(paths: List[str], crop_rect: Optional[CropRect] = None, 
            transform: Optional[TransformParams] = None, func_transform: Optional[TransformFunc] = None) -> torch.Tensor:
        imgs = []
        for path in paths:
            if path is not None:
                img = np.load(path)["arr_0"]

                if crop_rect:
                    img = img[:, crop_rect.y1: crop_rect.y2, crop_rect.x1: crop_rect.x2]
                if transform and func_transform:
                    img = func_transform(img, transform)
                    img = np.ascontiguousarray(img)

                img = torch.from_numpy(img.copy()).float()
                imgs.append(img)
        
        return torch.stack(imgs, dim=0)

    @staticmethod
    def _get_random_transform_params(hflip: bool = True, vflip: bool = True, rotation: bool = True, ratio: float = 0.5) -> TransformParams:
        return TransformParams(
            hflip and random.random() < ratio, 
            vflip and random.random() < ratio, 
            rotation and random.random() < ratio
        )

    @staticmethod
    def _fn_transform(img: np.ndarray, params: TransformParams) -> np.ndarray:
        if params.hflip:  # Horizontal flip.
            img = np.flip(img, 1)
        if params.vflip:  # Vertical flip.
            img = np.flip(img, 2)
        if params.rot90:  # Rotation 90.
            img = img.transpose(0, 2, 1)  # (x, y) -> (y, x)
        return img

    @staticmethod
    def _fn_transform_velocity(motion: np.ndarray, params: TransformParams) -> np.ndarray:
        if params.hflip:
            motion = np.flip(motion, 1)  # Flip in h dimension
            motion[1] = -motion[1]  # y = -y
        if params.vflip:
            motion = np.flip(motion, 2)  # Flip in w dimension
            motion[0] = -motion[0]  # x = -x
        if params.rot90:
            motion = motion.transpose(0, 2, 1)  # (x, y) -> (y, x)
            motion = motion[::-1]
        return motion
    
    @property
    def scene_name(self):
        return self._scene
