def validate_params_binary(func):
    def wrapper(params, original_dict):
        for key, value in params.items():
            if 'pred_class' not in value.keys():
                raise ValueError(f"Parameter '{key}' is missing 'pred_class' key.")
            if int(value['pred_class']) not in [0, 1]:
                raise ValueError(f"'pred_class' value for parameter '{key}'  must be 0 or 1.")
        return func(params, original_dict)
    return wrapper