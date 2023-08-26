import functools
import math
import operator
import re
import warnings

import numpy as np

def validate_params_binary(func):
    def wrapper(params, original_dict):
        for key, value in params:
            if 'pred_class' not in value:
                raise ValueError(f"Parameter '{key}' is missing 'pred_class' key.")
            if value['pred_class'] not in [0.0, 1.0]:
                raise ValueError(f"'pred_class' value for parameter '{key}'  must be 0 or 1.")
            return func(params, original_dict)
    return wrapper