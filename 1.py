# import os
# import sys
# import enum
# from datetime import datetime
# from io import TextIOWrapper
# from typing import Optional

# __all__ = ["record", "info", "warning", "error", "flush"]


# class _log_level(enum.Enum):
#     RECORD = enum.auto()
#     INFO = enum.auto()
#     WARNING = enum.auto()
#     ERROR = enum.auto()


# class _logger(object):
#     def __new__(cls, *args, **kwargs):
#         if not hasattr(cls, "__unique_instance"):
#             cls.__unique_instance = super().__new__(cls)
#         return cls.__unique_instance
    
#     def __init__(self):
#         print('__init__方法被调用')

#         date = datetime.now().strftime("%Y-%m-%d")
#         log_file_path = f".logs/{date}/{datetime.now()}.log".replace(":", "-")
#         os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
#         self._log_file: Optional[TextIOWrapper] = None
#         if sys.gettrace() is not None:
#             self._log_file = open(log_file_path, mode="w+")

#     def __del__(self):
#         print("__del__方法被调用")
#         if self._log_file is not None:
#             self._log(_log_level.INFO, "log file is closed")
#             self._flush()
#             if not self._log_file.closed:
#                 self._log_file.close()
        
#     def _log(self, level: _log_level, *message: str, sep: str, end: str = "\n"):
#         basic_info = "[%s][%s]" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level.name)
#         log_msg = sep.join(message)
        
#         if self._log_file is not None:
#             self._log_file.write("%s %s%s" % (basic_info, log_msg, end))
        
#         if level is _log_level.INFO:
#             print("\033[32m%s %s\033[0m" % (basic_info, log_msg), end=end)
#         elif level is _log_level.WARNING:
#             print("\033[33m%s %s\033[0m" % (basic_info, log_msg), end=end)
#         elif level is _log_level.ERROR:
#             print("\033[31m%s %s\033[0m" % (basic_info, log_msg), end=end)
    
#     def _flush(self):
#         if self._log_file is not None:
#             self._log_file.flush()


# cat1 = _logger()
# cat2 = _logger()

# print(id(cat1) == id(cat2))

from core.tools import logger
logger.release()