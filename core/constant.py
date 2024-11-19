import sys

ENABLE_DEBUG_TRACE = sys.gettrace() is not None
ENABLE_CUDNN = bool(False)

LOG_LOSS_INTERVAL = int(5)

ENABLE_PIXEL_CLIP = bool(True)
CLIP_MIN_PIXEL = float(0)
CLIP_MAX_PIXEL = float(10)
GAMMA_CORRECTION = float(2.2)
MAX_HISTORY_FRAMES = int(4)

CLIP_GRAD = float(10)
GLOBAL_FORWARD_COUNT = int(0)
