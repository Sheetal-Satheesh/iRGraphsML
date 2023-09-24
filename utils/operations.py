from collections import OrderedDict, Counter
from rdflib import URIRef
from utils.concept_generator import ConceptGenerator
import pandas as pd


def remove_nested_list(input_list, target_list):
    # Remove elements from the input_list that are present in the target_list.
    return [item for item in input_list if item not in target_list]


def delete_nested_list_from_dict(dictionary, key, nested_list_to_remove):
    if key in dictionary:
        dictionary[key] = [nested_list for nested_list in dictionary[key] if nested_list != nested_list_to_remove]


def find_disjoint_lists(patterns):
    """
        Returns mutually exclusive patterns for each key in the input dictionary.

        This function takes a dictionary 'patterns' with keys and nested lists as values.
        It identifies mutually exclusive patterns for each key, where the patterns in the values of each key
        do not have any items in common with the patterns in the values of other keys.

        Parameters:
            patterns (dict): A dictionary with keys as identifiers and values as nested lists of patterns.

        Returns:
            dict: A new dictionary where each key corresponds to the original keys, and the values are lists
            containing mutually exclusive patterns associated with each key.

        Example:
            patterns = {
                'A': [[1, 2], [3, 4], [5, 6]],
                'B': [[7, 8], [1, 2], [9, 10], [11, 12]],
                'C': [[13, 14], [11, 12], [15, 16], [17, 18]]
            }
            result = find_disjoint_lists(patterns)
            print(result)
            # Output: {'A': [[3, 4], [5, 6]], 'B': [[7, 8], [9, 10]], 'C': [[13, 14], [15, 16], [17, 18]]}
        """
    mutually_exclusive_dict = {}
    keys_copy = list(patterns.keys())

    for key in keys_copy:
        lst = patterns[key]
        first_list = lst[:]
        list_after_removing = []
        items_to_remove = []
        flag = False

        new_k = patterns.keys()
        # check if any item is there in any other list
        if len(new_k) > 1:
            for next_key, chk_lst in patterns.items():
                if next_key != key:
                    list_to_compare = chk_lst[:]
                    for items in first_list:
                        if any(Counter(items) == Counter(sublist) for sublist in list_to_compare):
                            flag = True
                            items_to_remove.append(items)
                            patterns[next_key] = [nested_list for nested_list in patterns[next_key] if
                                                  nested_list != items]
                    if not flag:
                        items_to_remove = list_to_compare
            if items_to_remove:
                list_after_removing = remove_nested_list(first_list, items_to_remove)
        elif len(new_k) == 1:
            for k, v in patterns.items():
                list_after_removing = v

        if key not in mutually_exclusive_dict:
            mutually_exclusive_dict[key] = list_after_removing

        del patterns[key]

    return mutually_exclusive_dict


def get_pattern_list(pattern: list, length: int = 3) -> list:
    resultant_pattern = []
    for item in pattern:
        if len(item) > length:
            concept_generator = ConceptGenerator(item)
            resultant_pattern.append(concept_generator.generate_path())
    return resultant_pattern


def remove_uri_from_dict(uri_repr, length=2):
    removed_uri = {}
    for key, v in uri_repr.items():
        if key not in removed_uri:
            removed_uri[key] = [get_pattern_list(v[0], length=length), v[1]]
    return removed_uri


def remove_uri_prefix(uri):
    if isinstance(uri, URIRef):
        return str(uri).split("#")[-1]
    return str(uri)


def merge_dict(input_dict):
    merged_dict = {}
    for inner_dict in input_dict.values():
        merged_dict.update(inner_dict)
    return merged_dict

def filter_common_patterns(patterns_by_class):
    filtered_patterns = {}  # class -> list of patterns

    for class_label, class_patterns in patterns_by_class.items():
        filtered_patterns[class_label] = []

        for pattern, freq in class_patterns:
            max_freq = freq

            # Check against patterns in other classes
            for other_class_label, other_class_patterns in patterns_by_class.items():
                if other_class_label != class_label:
                    other_class_freq = next((item[1] for item in other_class_patterns if item[0] == pattern), 0)
                    max_freq = max(max_freq, other_class_freq)

            if freq == max_freq:
                filtered_patterns[class_label].append((pattern, freq))

    return filtered_patterns

def remove_frequency_count(pattern_by_labels):
    clean_patterns = {}

    for class_label, class_patterns in pattern_by_labels.items():
        clean_patterns[class_label] = [item[0] for item in class_patterns]

    return clean_patterns

def get_most_occurring_pattern_for_random_walks(random_walk, frequency_count=False, count=2):
    most_occurring_pattern = {}
    if bool(random_walk):
        unique_labels = random_walk.keys()
        for cls in unique_labels:
            pattern = check_most_occurring_pattern(random_walk[cls], frequency_count=frequency_count,
                                                        count=count)
            if cls not in most_occurring_pattern:
                most_occurring_pattern[cls] = pattern
        return most_occurring_pattern
    else:
        raise Exception('Random Walks must be performed to find the most frequent pattern.')

def check_most_occurring_pattern(patterns, frequency_count=False, count=2):
    # Count the occurrences of each unique list
    counts = Counter(tuple(sublist) for sublist in patterns)

    # Find the list with the maximum count
    most_common_list = max(counts, key=counts.get)
    if frequency_count:
        most_occuring_random_walk_occurrences = [item for item in counts.most_common(count)]
    else:
        most_occuring_random_walk_occurrences = [item[0] for item in counts.most_common(count)]
    return most_occuring_random_walk_occurrences

def format_data_for_metrics(test_data=None, predicted_data=None):
        test_ids = []
        actual_predictions = []
        predicted_predictions = []

        # Iterate through test data
        for class_label, inner_dict in test_data.items():
            for test_id, _ in inner_dict.items():
                test_ids.append(test_id)
                actual_predictions.append(class_label)

                # Match test_id with predicted_data
                predicted_label = None
                for pred_id, inner_pred_dict in predicted_data.items():
                    if test_id == pred_id:
                        predicted_label = inner_pred_dict['label']
                        break

                predicted_predictions.append(predicted_label)

        # Create a DataFrame
        df = pd.DataFrame({
            'testid': test_ids,
            'actual_prediction': actual_predictions,
            'predicted_prediction': predicted_predictions
        })
        df.to_csv('xyz.csv')
        return df