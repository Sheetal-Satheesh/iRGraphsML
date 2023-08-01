from collections import OrderedDict, Counter
from rdflib import URIRef
from utils.concept_generator import ConceptGenerator


def remove_nested_list(input_list, target_list):
    return [item for item in input_list if item not in target_list]


def delete_nested_list_from_dict(dictionary, key, nested_list_to_remove):
    if key in dictionary:
        dictionary[key] = [nested_list for nested_list in dictionary[key] if nested_list != nested_list_to_remove]


def find_disjoint_lists(patterns):
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
            removed_uri[key] = get_pattern_list(v, length=length)
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
