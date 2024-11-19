import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import ctypes
import numpy as np

if sys.platform == "win32":
    __dll = ctypes.cdll.LoadLibrary(__file__.replace("__init__.py", "bin/color_converter.dll"))
elif sys.platform == "linux":
    __dll = ctypes.cdll.LoadLibrary(__file__.replace("__init__.py", "build/libcolor_converter.so"))
else:
    raise NotImplementedError("Platform not supported")

def convert_linear_to_srgb(image: np.ndarray): # bgr.
    assert image.ndim == 3 and image.shape[2] in [1, 3, 4], "The input image must be GRAY, RGB or RGBA."

    channels = image.shape[2]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    
    h, w = image.shape[:2]

    in_linear= image.reshape(w * h * 4)
    in_linear = in_linear.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    out_srgb = np.zeros_like(image, dtype=np.uint8)
    out_srgb = out_srgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    __dll.convert_linear_to_srgb(out_srgb, in_linear, w * h * 4, 255)
    out_srgb = np.array(out_srgb._arr)

    if channels == 3:
        image = cv2.cvtColor(out_srgb, cv2.COLOR_BGRA2BGR)
    elif channels == 1:
        image = cv2.cvtColor(out_srgb, cv2.COLOR_BGRA2GRAY)

    return out_srgb


def load_exr(path: str, bgr2rgb: bool = False, hwc2chw: bool = False) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if hwc2chw:
        image = image.transpose(2, 0, 1)
    return image


