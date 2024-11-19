import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..tools import logger

# Type Hints
__PadParams = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
PadParams = Tuple[__PadParams, __PadParams]

__CropParams = Tuple[int, int, int, int]
CropParams = Tuple[__CropParams, __CropParams]

TransformParams = Tuple[bool, bool, bool]
TransformFunc = Callable[[np.ndarray, TransformParams], np.ndarray]

# Constants
_MAX_HISTORY_FRAMES = int(4)


def check_nparray(path: str) -> str:
    assert os.path.exists(path) and os.path.splitext(path)[-1].lower() in [".npy", ".npz"], \
        "The file does not exist or is not a valid image array: %s" % path
    return path


def load_nparray(path: str) -> np.ndarray:
    return np.load(path)["arr_0"]


def augment(data: np.ndarray, crop_params: CropParams, pad_params: PadParams, pad_mode: str, 
            transform_params: TransformParams, transform_fn: TransformFunc) -> torch.Tensor:
    if crop_params:
        data = data[:, crop_params[2]: crop_params[3], crop_params[0]: crop_params[1]]
    if pad_params:
        data = np.pad(data, pad_params, mode=pad_mode)
    if transform_params:
        data = transform_fn(data, transform_params)
        data = np.ascontiguousarray(data)

    data = torch.from_numpy(data.copy()).float()
    return data


class SuperSamplingDataset(Dataset):
    def __init__(self, hr_frame_root: str, lr_frame_root: str, require_aux_types: Optional[Dict[str, bool]] = None,
                 hr_buffer_root: str = "", lr_buffer_root: str = "", velocity_root: str = "", # Auxiliary data: G-buffers or history info.
                 n_history_frames: int = 0,  n_frames: int = -1, frame_range: Optional[List[List[int]]] = None,
                 sr_scale: int = -1, lr_size: int = -1, fixed_lr_pad_size: int | List[int] = 0, 
                 phase: str = "train", scene: str = "Unknown") -> None:
        super().__init__()
        self._hr_frame_root = hr_frame_root
        self._lr_frame_root = lr_frame_root

        self._hr_buffer_root = hr_buffer_root
        self._lr_buffer_root = lr_buffer_root
        self._motion_vector_root = velocity_root

        if require_aux_types is None:
            require_aux_types = dict()
        self._require_aux_types = require_aux_types

        self._n_history_frames = n_history_frames
        self._parse_require_types(require_aux_types)

        self._phase = phase
        self._scene = scene
        if phase == "train":
            assert lr_size > 0 and sr_scale > 0, "lr_size and sr_scale must be set in training phase"

        self._sr_scale = sr_scale
        self._lr_size = lr_size

        if isinstance(fixed_lr_pad_size, int):
            fixed_lr_pad_size = [fixed_lr_pad_size, fixed_lr_pad_size]
        self._fixed_lr_pad_size = fixed_lr_pad_size

        data_table = self._build_data_table()

        if n_frames == -1 and frame_range is None:
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
        logger.info("[Data]", "Create %s dataset. Dataset size: %d, lr_size: %d, sr_scale: %d" % (phase, self._len, lr_size, sr_scale))
        logger.info("Min frame: %s. \n\tMax frame: %s" % (self._data_table[0]["lr"], self._data_table[-1]["lr"]))

        # Log data ranges.
        os.makedirs(".log/.temp", exist_ok=True)
        with open(".log/.temp/data_all.txt", "w") as all_data,    \
            open(".log/.temp/data_valid.txt", "w") as valid_data, \
            open(".log/.temp/data_train.txt", "w") as train_data:   
            for data_row in data_table:
                all_data.write(data_row["lr"] + "\n")
                if data_row in self._data_table:
                    train_data.write(data_row["lr"] + "\n")
                else:
                    valid_data.write(data_row["lr"] + "\n")

    def _parse_require_types(self, require_types: Dict[str, bool]):
        self._require_hr_buffer = require_types.get("hr_buffer", False)
        self._require_lr_buffer = require_types.get("lr_buffer", False)

        self._require_history_frames = require_types.get("history_frames", False)
        self._require_motion_vectors = require_types.get("motion_vectors", False) or require_types.get("velocity", False)
        self._require_history_buffer = require_types.get("history_buffer", False)

        # Check history info dependencies.
        if self._require_history_frames:
            assert self._require_motion_vectors, "History frames requires motion vectors."
            assert self._n_history_frames != 0 and self._n_history_frames <= _MAX_HISTORY_FRAMES, "Invalid history frames number."
        if self._require_history_buffer:
            assert self._require_history_frames, "History buffer requires history frames."            

        self._require_basename = require_types.get("basename", False)

    @staticmethod
    def _get_frame_index(frame_name: str) -> tuple[int, str, int]:
        # Image nale format: ScaneName.FrameIndex.RenderPass.Extension. Example: RedwoodForeast.0020.FinalImage.exr.
        frame_idx_str = frame_name.split(".")[-3]
        frame_idx_str_width = len(frame_idx_str)
        frame_idx = int(frame_idx_str)
        return frame_idx, frame_idx_str, frame_idx_str_width

    def _has_enough_history_frames(self, frame_name: str) -> bool:
        frame_idx, frame_idx_str, frame_idx_str_width = self._get_frame_index(frame_name)
        for i in range(1, max(self._n_history_frames, _MAX_HISTORY_FRAMES) + 1):
            path = os.path.join(self._lr_frame_root, frame_name.replace(frame_idx_str, str(frame_idx - i).zfill(frame_idx_str_width)))
            if not os.path.exists(path):
                return False
        return True
        
    def _build_data_table(self) -> List[Dict[str, Union[str, List[str]]]]:
        data_table = []
        
        for frame_name in sorted(os.listdir(self._lr_frame_root)):
            frame_idx, frame_idx_str, frame_idx_str_width = self._get_frame_index(frame_name)

            if self._require_history_frames and not self._has_enough_history_frames(frame_name):
                continue

            # Add a valid data row.
            data_row = {}
            data_row["hr"] = check_nparray(os.path.join(self._hr_frame_root, frame_name))
            data_row["lr"] = check_nparray(os.path.join(self._lr_frame_root, frame_name))

            if self._require_hr_buffer:
                data_row["hr_bf"] = check_nparray(os.path.join(self._hr_buffer_root, frame_name))
            if self._require_lr_buffer:
                data_row["lr_bf"] = check_nparray(os.path.join(self._lr_buffer_root, frame_name))

            if self._require_history_frames:
                # Add motion vector paths and history paths, history G-buffers is low resolution.
                data_row["motion_vectors"] = []
                data_row["history_frames"] = []
                
                if self._require_history_buffer:
                    data_row["history_buffer"] = []

                _idx_str = frame_idx_str
                for i in range(1, self._n_history_frames + 1):
                    # Add motion vectors path - v_i.
                    path = check_nparray(os.path.join(self._motion_vector_root, frame_name.replace(frame_idx_str, _idx_str)))
                    data_row["motion_vectors"].append(path)

                    # Add history frame path - f_{i-1}.
                    if frame_idx - i >= 0:
                        _idx_str = str(frame_idx - i).zfill(frame_idx_str_width)
                    path = check_nparray(os.path.join(self._lr_frame_root, frame_name.replace(frame_idx_str, _idx_str)))
                    data_row["history_frames"].append(path)
                
                    # Add history G-buffers path - f_{i-1}.
                    if self._require_history_buffer:
                        path = check_nparray(os.path.join(self._lr_buffer_root, frame_name.replace(frame_idx_str, _idx_str)))
                        data_row["history_buffer"].append(path)
                    
            if self._require_basename:
                data_row["basename"] = os.path.splitext(frame_name)[0]
            
            # Append a row to data table.
            data_table.append(data_row)

        return data_table
    
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], str]]:
        data_row = self._data_table[index]

        hr_frame = load_nparray(data_row["hr"])
        lr_frame = load_nparray(data_row["lr"])

        lr_pad, hr_pad, lr_crop, hr_crop, transform_params = None, None, None, None, None
        if self._phase == "train":
            lr_h, lr_w = lr_frame.shape[1], lr_frame.shape[2]
            transform_params = self._get_random_transform_params()
            if self._should_crop(lr_w, lr_h):
                lr_crop, hr_crop = self._get_random_crop_params(lr_w, lr_h)
            if self._should_pad(lr_w, lr_h):
                lr_pad, hr_pad = self._get_pad_params(lr_w, lr_h)
        elif self._fixed_lr_pad_size[0] > 0 or self._fixed_lr_pad_size[1] > 0:
            assert self._phase == "test", "Fixed pad size is only available for test phase."
            lr_pad, hr_pad = self._get_fixed_pad_params()

        hr_frame = augment(hr_frame, hr_crop, hr_pad, "reflect", transform_params, self._fn_transform)
        lr_frame = augment(lr_frame, lr_crop, lr_pad, "reflect", transform_params, self._fn_transform)
        return_data = [hr_frame, lr_frame]

        if self._require_hr_buffer:
            hr_buffer = self._load_and_augment_single(
                data_row["hr_bf"], hr_crop, hr_pad, "reflect", transform_params, self._fn_transform)
            return_data.append(hr_buffer)
        if self._require_lr_buffer:
            lr_buffer = self._load_and_augment_single(
                data_row["lr_bf"], lr_crop, lr_pad, "reflect", transform_params, self._fn_transform)
            return_data.append(lr_buffer)

        if self._require_history_frames:
            motion_vectors = self._load_and_augment_multiple(
                data_row["motion_vectors"], lr_crop, lr_pad, "reflect", transform_params, self._fn_transform_velocity)
            return_data.append(motion_vectors)
            
            history_frames = self._load_and_augment_multiple(
                data_row["history_frames"], lr_crop, lr_pad, "reflect", transform_params, self._fn_transform)
            return_data.append(history_frames)

            if self._require_history_buffer:
                history_buffer = self._load_and_augment_multiple(
                    data_row["history_buffer"], lr_crop, lr_pad, "reflect", transform_params, self._fn_transform)
                return_data.append(history_buffer)
        
        if self._require_basename:
            return return_data, data_row["basename"]
        else:
            return return_data
    
    @staticmethod
    def _load_and_augment_single(path: str, crop_params: CropParams, pad_params: PadParams, pad_mode: str, 
                                 transform_params: TransformParams, transform_fn: TransformFunc) -> torch.Tensor:
        data = load_nparray(path)
        data = augment(data, crop_params, pad_params, pad_mode, transform_params, transform_fn)
        return data

    def _load_and_augment_multiple(self, paths: List[str], crop_params: CropParams, pad_params: PadParams, pad_mode: str, 
                                   transform_params: TransformParams, transform_fn: TransformFunc) -> torch.Tensor:
        return torch.stack([
            self._load_and_augment_single(path, crop_params, pad_params, pad_mode, transform_params, transform_fn) 
            for path in paths
        ])

    def _get_random_transform_params(hflip: bool = True, vflip: bool = True, rotation: bool = True, ratio: float = 0.5) -> TransformParams:
        hflip = hflip and random.random() < ratio
        vflip = vflip and random.random() < ratio
        rot90 = rotation and random.random() < ratio
        return hflip, vflip, rot90

    def _should_crop(self, lr_w: int, lr_h: int):
        return lr_w > self._lr_size or lr_h > self._lr_size
   
    def _should_pad(self, lr_w: int, lr_h: int):
        return (lr_w < self._lr_size or lr_h < self._lr_size)

    def _get_random_crop_params(self, lr_w: int, lr_h: int) -> CropParams:
        lr_size = self._lr_size
        lr_y = random.randint(0, lr_h - lr_size)
        lr_x = random.randint(0, lr_w - lr_size)       

        hr_size = self._lr_size * self._sr_scale
        hr_y = lr_y * self._sr_scale
        hr_x = lr_x * self._sr_scale
        return (lr_x, lr_x + lr_size, lr_y, lr_y + lr_size), (hr_x, hr_x + hr_size, hr_y, hr_y + hr_size)
    
    def _get_pad_params(self, lr_w: int, lr_h: int) -> PadParams:
        pad_w = max(0, self._lr_size - lr_w)
        pad_h = max(0, self._lr_size - lr_h)
        lr_pad = ((0, 0), (pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2))
        hr_pad = ((0, 0), (pad_h // 2 * self._sr_scale, (pad_h + 1) // 2 * self._sr_scale), 
                          (pad_w // 2 * self._sr_scale, (pad_w + 1) // 2 * self._sr_scale))
        return lr_pad, hr_pad
    
    def _get_fixed_pad_params(self) -> PadParams:
        lr_pad = ((0, 0), (0, self._fixed_lr_pad_size[1]), (0, self._fixed_lr_pad_size[0]))
        hr_pad = ((0, 0), (0, self._fixed_lr_pad_size[1] * self._sr_scale), (0, self._fixed_lr_pad_size[0] * self._sr_scale))
        return lr_pad, hr_pad

    @staticmethod
    def _fn_transform(img: np.ndarray, params: TransformParams) -> np.ndarray:
        if params[0]:  # Horizontal flip.
            img = np.flip(img, 1)
        if params[1]:  # Vertical flip.
            img = np.flip(img, 2)
        if params[2]:  # Rotation 90.
            img = img.transpose(0, 2, 1)  # (x, y) -> (y, x)
        return img

    @staticmethod
    def _fn_transform_velocity(motion: np.ndarray, params: TransformParams) -> np.ndarray:
        if params[0]:
            motion = np.flip(motion, 1)  # Flip in h dimension
            motion[1] = -motion[1]  # y = -y
        if params[1]:
            motion = np.flip(motion, 2)  # Flip in w dimension
            motion[0] = -motion[0]  # x = -x
        if params[2]:
            motion = motion.transpose(0, 2, 1)  # (x, y) -> (y, x)
            motion = motion[::-1]
        return motion
    
    @property
    def scene_name(self):
        return self._scene

    @property
    def fixed_pad_size(self):
        if self._fixed_lr_pad_size[0] > 0 or self._fixed_lr_pad_size[1] > 0:
            return self._get_fixed_pad_params()
        return None
    