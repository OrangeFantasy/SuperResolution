import os
import sys
import enum
from datetime import datetime
from io import TextIOWrapper
from typing import Optional

__all__ = ["record", "info", "warning", "error", "flush"]


class _log_level(enum.Enum):
    RECORD = enum.auto()
    INFO = enum.auto()
    WARNING = enum.auto()
    ERROR = enum.auto()


class _logger(object):
    def __new__(cls):
        if not hasattr(cls, "__unique_instance"):
            cls.__unique_instance = super(_logger, cls).__new__(cls)
        return cls.__unique_instance
    
    def __init__(self) -> None:
        print("__init__")
        date = datetime.now().strftime("%Y-%m-%d")
        log_file_path = f".logs/{date}/{datetime.now()}.log".replace(":", "-")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        self._log_file: Optional[TextIOWrapper] = None
        if sys.gettrace() is not None or True:
            self._log_file = open(log_file_path, mode="w+")
    
    def __del__(self):
        print("__del__")
        if self._log_file is not None:
            self._flush()
            if not self._log_file.closed:
                self._log_file.close()

    def _log(self, level: _log_level, *message: str, sep: str, end: str = "\n"):
        basic_info = "[%s][%s]" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level.name)
        log_msg = sep.join(message)
        
        if self._log_file is not None:
            self._log_file.write("%s %s%s" % (basic_info, log_msg, end))
        
        if level is _log_level.INFO:
            print("\033[32m%s %s\033[0m" % (basic_info, log_msg), end=end)
        elif level is _log_level.WARNING:
            print("\033[33m%s %s\033[0m" % (basic_info, log_msg), end=end)
        elif level is _log_level.ERROR:
            print("\033[31m%s %s\033[0m" % (basic_info, log_msg), end=end)
    
    def _flush(self):
        if self._log_file is not None:
            self._log_file.flush()


__logger = _logger()


def record(*message, sep: str = " "):
    __logger._log(_log_level.RECORD, *message, sep=sep)


def info(*message, sep: str = " "):
    __logger._log(_log_level.INFO, *message, sep=sep)


def warning(*message, sep: str = " "):
    __logger._log(_log_level.WARNING, *message, sep=sep)


def error(*message, sep: str = " "):
    __logger._log(_log_level.ERROR, *message, sep=sep)


def record_dict(*message, dict_msg: dict, sep: str = " "):
    msg = sep.join(message) + "\n"

    def _format_dict(_all_msg: dict, _prefix: str) -> str:
        _msg = ""
        for _key, _value in _all_msg.items():
            _msg += _prefix + _key + ": " 
            if isinstance(_value, dict):
                _msg += "\n"
                _msg += _format_dict(_value, _prefix + "  ")
            else:
                _msg += str(_value) + "\n"
        return _msg
    
    msg += _format_dict(dict_msg, "  ")
    __logger._log(_log_level.RECORD, msg, sep="", end="")


def flush():
    __logger._flush()


def release():
    __logger.__del__()
    