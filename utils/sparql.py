from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
from collections import OrderedDict
import copy

class DecodeRWSparQL:
    def __init__(self, path_sequence, graph):
        self.path = OrderedDict(path_sequence)
        self.graph = graph

    def predict(self, test_data):
        # TODO: Add proper comments
        predicted_dict = {}
        test_ids = []
        for keys, val in test_data.items():
            test_ids.extend(test_data[keys].keys())

        no_of_classes = len(list(self.path.keys()))
        class_labels = list(self.path.keys())
        i = 0
        while i < no_of_classes:
            key = class_labels[i]
            v = self.path[key]
            for test_id in test_ids:
                sub_paths = [list(nested_tuple) for nested_tuple in v]
                is_a_class, path = self.predict_class(sub_paths, test_id)
                if is_a_class:
                    if test_id not in predicted_dict:
                        predicted_dict[test_id] = {
                            'pred_class': key,
                            'path': path
                        }
                    else:
                        value = predicted_dict[test_id]['pred_class']
                        if value is None:
                            predicted_dict[test_id] = {
                            'pred_class': key,
                            'path': path
                        }
                else:
                    if test_id not in predicted_dict:
                        predicted_dict[test_id] = {
                            'pred_class': None,
                            'path': path
                        }
            i = i + 1

        new_value = class_labels[-1]
        for key, _ in predicted_dict.items():
            if predicted_dict[key]['pred_class'] is None:
                predicted_dict[key]['pred_class'] = new_value
        return predicted_dict


    def predict_class(self, paths, test_id):
        flag = False
        for sub_path in paths:
            if len(sub_path) >= 3:
                path = copy.deepcopy(sub_path)
                sub_type = sub_path.pop(0)
                pred = sub_path.pop(0)
                obj_type = sub_path.pop(0)
                results = self.generate_sparql_query_with_id_and_type(test_id, sub_type, pred, obj_type)
                l_ids = [row.res for row in results if row.res is not None]

                if len(l_ids) != 0 and len(sub_path) != 0:
                    is_valid_path = self.check_path_validity(sub_path, l_ids)
                    if is_valid_path:
                        flag = True
                        return flag, path
                elif len(l_ids) != 0 and len(sub_path) == 0:
                    flag = True
                    return flag, path
        return flag, None

    
    def check_path_validity(self, path, list_ids):
        pred_type = path.pop(0)
        obj_type = path.pop(0)
        flag = False
        for id in list_ids:
            results = self.generate_sparql_query_with_id(id, pred_type, obj_type)
            value_res = [row.res for row in results if row.res is not None]
            if len(path) != 0 and len(value_res) == 0:
                continue
            elif len(path) == 0 and len(value_res) != 0:
                return True
            elif len(path) == 0 and len(value_res) == 0:
                continue
            else:
                flag = self.check_path_validity(path, value_res)
        return flag


    def generate_sparql_query_with_id_and_type(self, bond_id, sub_type, pred, obj_type):
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?res
            WHERE {{
                <{bond_id}> rdf:type <{sub_type}> .
                <{bond_id}> <{pred}> ?res . 
                ?res rdf:type <{obj_type}>
            }}
        """
        results = self.graph.query(query)
        return results


    def generate_sparql_query_with_id(self, sub_type, pred, obj_type):
        query_string = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?res
        WHERE {{
                {{                
                    <{sub_type}> <{pred}> ?res  .
                    FILTER (
                        isLiteral(?res) &&
                        (datatype(?res) = xsd:double || datatype(?res) = xsd:float || datatype(?res) = xsd:int || datatype(?res) = xsd:boolean) && 
                        datatype(?res) = <{obj_type}>
                    )
                }}            
            UNION 
            {{       
                <{sub_type}> <{pred}> ?res  .
                ?res rdf:type <{obj_type}> .
                FILTER (!isLiteral(?res))
            }}
        }}
        """

        results = self.graph.query(query_string)
        return results