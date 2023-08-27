def validate_params_binary(func):
    def wrapper(params, original_dict):
        for key, value in params.items():
            if 'label' not in value.keys():
                raise ValueError(f"Parameter '{key}' is missing 'label' key.")
            if int(value['label']) not in [0, 1]:
                raise ValueError(f"'label' value for parameter '{key}'  must be 0 or 1.")
        return func(params, original_dict)
    return wrapper