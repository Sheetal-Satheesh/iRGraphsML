from utils._param_validation import validate_params_binary

def recall(params, original_dict):
    try:
        tp, tn, fp, fn = score(params, original_dict)
        recall = tp / (tp + fn)
        return round(recall, 3)
    except ZeroDivisionError:
        return 0


def precision(params, original_dict):
    try:
        tp, tn, fp, fn = score(params, original_dict)
        precision = tp / (tp + fp)
        return round(precision, 3)
    except ZeroDivisionError:
        return 0


def f1_score(params, original_dict):
    try:
        tp, tn, fp, fn = score(params, original_dict)
        recall_value = tp / (tp + fn)
        precision_value = tp / (tp + fp)

        if precision_value == 0 or recall_value == 0:
            return 0

        f1_score = 2 * ((precision_value * recall_value) / (precision_value + recall_value))
        return f1_score
    except ZeroDivisionError:
        return 0

def accuracy(params, original_dict):
    try:
        tp, tn, fp, fn = score(params, original_dict)
        acc = (tp + tn) / (tp + tn + fp + fn)
        return acc
    except ZeroDivisionError:
        return 0

@validate_params_binary
def score(params, original_dict):
    """
    :param params:
    :param original_dict:
    :return: count of tp, tn, fp, fn
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for key, value in original_dict.items():
        original_values = list(value.keys())
        original_class = int(key)
        for k in params:
            predicted_class = int(params[k]['label'])
            predicted_value = k
            if predicted_value in original_values:
                if original_class == 1 and predicted_class == 1:
                    tp += 1
                elif original_class == 0 and predicted_class == 0:
                    tn += 1
                elif original_class == 0 and predicted_class == 1:
                    fp += 1
                elif original_class == 1 and predicted_class == 0:
                    fn += 1
    return tp, tn, fp, fn