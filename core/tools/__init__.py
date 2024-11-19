import os
import importlib
from omegaconf import OmegaConf
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.nn import functional as F


# def parse_argument(param: str):
#     key, value = param.split("=", maxsplit=1)
#     if value.isdigit():
#         value = int(value)
#     elif value.replace(".", "", 1).replace("e", "", 1).isdigit():
#         value = float(value)
#     elif value == "True" or value == "true":
#         value = True
#     elif value == "False" or value == "false":
#         value = False
#     elif value == "None" or value == "none":
#         value = None
#     elif value.startswith("{") and value.endswith("}"):
#         _params = {}
#         for param in value[1:-1].split(";"):
#             _key, _value = parse_argument(param)
#             _params[_key] = _value
#         value = _params
#     else:
#         raise ValueError(f"Unsuported argument type: {value}")
#     return key, value
    

# def parse_args(self, args=None, namespace=None):
#     args, argv = self.parse_known_args(args, namespace)
#     if argv:
#         msg = _('unrecognized arguments: %s')
#         self.error(msg % ' '.join(argv))
#     return args
import json

def parse_json_from_str(json_str: str) -> Dict[str, Any]:
    return json.loads(json_str)


# args_str like: {a=10; b=[12, 34, 74]; c="abc"; d={da=1; cd=2.3}}
# def parse_args(args_str: str, index):
#     args = {}

#     while index < len(args_str):
#         if args_str[index] == "{":
#             index += 1


    # assert args_str.startswith("{") and args_str.endswith("}")
#     args_str = args_str[1:-1]

#     idx = 0
#     while idx < len(args_str):
#         key_value = args_str.split(";", maxsplit=1)[0].strip()
#         if key_value.startswith("{") and key_value.endswith("}"):
#             key, value = parse_args(key_value)
#             args[key] = value
#         else:
#             key, value_str = key_value.split("=", maxsplit=1)
#             args[key] = parse_value(value_str)



# def parse_value(value_str: str):
#     if value_str.isdigit():
#         return int(value_str)
#     elif value_str.replace(".", "", 1).replace("e", "", 1).isdigit():
#         return float(value_str)
#     elif value_str == "True" or value_str == "true":
#         return True
#     elif value_str == "False" or value_str == "false":
#         return False
#     elif value_str == "None" or value_str == "none":
#         return None
#     elif value_str.startswith("[") and value_str.endswith("]"):
#         return [parse_value(v) for v in value_str[1:-1].split(",")]
#     elif value_str.startswith("{") and value_str.endswith("}"):
#         return parse_args(value_str)
#     else:
#         raise ValueError(f"Unsuported argument type: {value_str}")



    # return args


# args = parse_json_from_str("{\"a\": 10}" b=[12, 34, 74], c=abc, d={da=1; cd=2.3}}")
# print(args)