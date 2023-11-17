import copy
from collections import OrderedDict

from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias
from utils.operations import find_disjoint_lists, get_most_occurring_pattern_for_random_walks
from rdflib import Literal, URIRef
import pandas as pd
from rdflib.plugins.sparql import sparql


class RDFPathClassifierWithSPARQl:
    """
        RDF Path Classifier using SPARQL queries.

        This class provides methods for training a classifier based on RDF paths generated using random walks
        and making predictions on test data using SPARQL queries.

        Args:
            graph (rdflib.Graph): An RDF graph containing the knowledge base.
            algorithm (RandomWalk): An optional preconfigured RandomWalk object for generating random walks.
            prominent_class (str): An optional class label.

        Attributes:
            graph (rdflib.Graph): The RDF graph containing the knowledge base.
            algorithm (RandomWalk): The RandomWalk object for generating random walks.
            prominent_class (str): The fallback class label for predictions.
            path_sequence (collections.OrderedDict): The sequence of mutually exclusive RDF path patterns.
            path (None or collections.OrderedDict): Deprecated. Use path_sequence instead.

        """
    def __init__(self, rdf_graph, algorithm=None, prominent_class=None):
        self.rdf_graph = rdf_graph
        self.algorithm = algorithm
        self.prominent_class = prominent_class
        self.path_sequence = None
        self.path = None


    def __str__(self):
        if self.path_sequence:
            # Print the top 4 path sequences for each class label
            output = "Top 4 Path Sequences:\n"
            for class_label, paths in self.path_sequence.items():
                output += f"Class Label: {class_label}\n"
                for i, path in enumerate(paths[:4], 1):
                    output += f"Path {i}: {' -> '.join(map(str, path))}\n"
                output += "\n"
            return output
        else:
            return "Path sequence not set. Please run fit() to generate path sequences."

    def set_path_sequence(self, data_dict, num_walks, depth):
        """
            Set the path sequence by generating random walks, processing paths, and finding patterns.

            Args:
                data_dict (dict): A dictionary containing data for random walks.
                num_walks (int): The number of random walks to generate.
                depth (int): The depth of the random walks.

        """
        rw1 = self._create_random_walk_object(data_dict, self.algorithm, num_walks, depth)

        # Generate random walks
        rw1.set_random_walk()
        paths_from_walks = rw1.get_random_walk()

        # Process paths based on class labels
        paths = self.process_paths_based_on_class_labels(paths_from_walks)

        # Find most occurring patterns
        most_occurring_pattern = get_most_occurring_pattern_for_random_walks(paths, frequency_count=False, count=30)

        # Find mutually exclusive patterns
        new_most_occurring_pattern = copy.deepcopy(most_occurring_pattern)
        mutually_exclusive_pattern_app_two = find_disjoint_lists(new_most_occurring_pattern)

        # Set path sequence
        self.path_sequence = OrderedDict(mutually_exclusive_pattern_app_two)

    def fit(self, train_data, num_walks, depth):
        """
        Fit the classifier by setting the path sequence and return it.

        Args:
            train_data (dict): A dictionary containing data for random walks.
            num_walks (int): The number of random walks to generate.
            depth (int): The depth of the random walks.

        Returns:
            collections.OrderedDict: The path sequence.

        """
        self.set_path_sequence(train_data, num_walks, depth)
        return self.path_sequence

    def predict(self, test_data):
        """
            Make predictions on test data using RDF paths and SPARQL queries.

            Args:
                test_data (dict): A dictionary containing test data.

            Returns:
                dict: A dictionary containing test IDs, predicted labels, and paths.

        """
        predicted_dict = {}
        test_ids = []

        # Collect test IDs
        for keys, val in test_data.items():
            test_ids.extend(test_data[keys].keys())

        class_labels = list(self.path_sequence.keys())
        print('Start Prediciting...')

        for key in class_labels:
            # if key != self.prominent_class:
            v = self.path_sequence[key]

            for test_id in test_ids:
                sub_paths = [list(nested_tuple) for nested_tuple in v]
                is_a_class, path = self.predict_class(sub_paths, test_id)
                if is_a_class:
                    if test_id not in predicted_dict:
                        predicted_dict[test_id] = {
                            'label': key,
                            'path': path
                        }
                    else:
                        value = predicted_dict[test_id]['label']
                        if value is None:
                            predicted_dict[test_id] = {
                                'label': key,
                                'path': path
                            }
                else:
                    if test_id not in predicted_dict:
                        predicted_dict[test_id] = {
                            'label': None,
                            'path': path
                        }

        new_value = self.prominent_class
        print('Prominent class:', self.prominent_class)

        # Replace None labels with prominent class
        for key, _ in predicted_dict.items():
            if predicted_dict[key]['label'] is None:
                predicted_dict[key]['label'] = new_value

        predicted_dict_with_removed_uri = self.replace_uris_in_path(predicted_dict)
        predictions, actual_labels = self.format_data_for_metrics(test_data, predicted_dict_with_removed_uri)

        return predictions, actual_labels

    def predict_class(self, paths, test_id):
        """
            Predict the class label for a test data point.

            Args:
                paths (list): A list of RDF paths.
                test_id (str): The test data identifier.

            Returns:
                tuple: A tuple containing a boolean indicating if a class was predicted and the predicted path.

        """
        flag = False
        for sub_path in paths:
            if len(sub_path) >= 3:
                path = copy.deepcopy(sub_path)
                sub_type = sub_path.pop(0)
                pred = sub_path.pop(0)
                obj = sub_path.pop(0)
                results = self.generate_sparql_query_with_id_and_type(test_id, sub_type, pred, obj)
                ask_answer = results.askAnswer
                # print(ask_answer)
                if ask_answer is not None:
                    if results:
                        flag = True
                        return flag, path
                    else:
                        continue

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
        """
            Check the validity of an RDF path for a list of data points.

            Args:
                path (list): A list representing an RDF path.
                list_ids (list): A list of data point identifiers.

            Returns:
                bool: True if the path is valid for any data point, otherwise False.

        """
        pred_type = path.pop(0)
        obj_type = path.pop(0)
        flag = False
        for item in list_ids:
            results = self.generate_sparql_query_with_id(item, pred_type, obj_type)
            ask_answer = results.askAnswer

            if ask_answer is not None:
                if results:
                    return True
                else:
                    continue

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

    def generate_sparql_query_with_id_and_type(self, id, sub_type, pred, obj):
        """
            Generate a SPARQL query to retrieve data based on identifiers and types.

            Args:
                id (str): The identifier for a bond.
                sub_type (str): The subject's RDF type.
                pred (str): The predicate.
                obj_type (str): The object's RDF type.

            Returns:
                rdflib.plugins.sparql.processor.SPARQLResult: SPARQL query results.

        """
        try:
            if isinstance(obj, Literal):
                if obj is None:
                    # Handle the case where obj_value is None
                    # You can choose to exclude it from the query or treat it as an empty string
                    query = f"""
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        ASK
                        WHERE {{
                            <{id}> rdf:type <{sub_type}> .
                            <{id}> <{pred}> "" .
                        }}
                    """
                elif obj == "":
                    # If the object is an empty string
                    query = f"""
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX owl: <http://www.w3.org/2002/07/owl#>
                            ASK
                            WHERE {{
                                <{id}> rdf:type <{sub_type}> .
                                <{id}> <{pred}> "" .
                            }}
                        """
                else:
                    # If the object is a literal, construct an ASK query
                    escaped_obj_value = obj.n3()
                    # Otherwise, construct the ASK query with the escaped object value
                    query = f"""
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX owl: <http://www.w3.org/2002/07/owl#>
                            ASK
                            WHERE {{
                                <{id}> rdf:type <{sub_type}> .
                                <{id}> <{pred}> {escaped_obj_value} .
                            }}
                        """
            else:
                query = f"""
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                        SELECT ?res
                        WHERE {{
                            <{id}> rdf:type <{sub_type}> .
                            <{id}> <{pred}> ?res . 
                            ?res rdf:type <{obj}>
                        }}
                    """
        except sparql.SPARQLExceptions.QueryBadFormed as e:
            print(f"Error in SPARQL query: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        results = self.rdf_graph.query(query)
        return results

    def generate_sparql_query_with_id(self, sub_type, pred, obj):
        """
            Generate a SPARQL query to retrieve data based on identifiers.

            Args:
                sub_type (str): The subject's RDF type.
                pred (str): The predicate.
                obj_type (str): The object's RDF type.

            Returns:
                rdflib.plugins.sparql.processor.SPARQLResult: SPARQL query results.

        """
        try:
            if isinstance(obj, Literal):
                if obj is None:
                    # Handle the case where obj_value is None
                    query = f"""
                         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                         PREFIX owl: <http://www.w3.org/2002/07/owl#>
                         ASK
                         WHERE {{
                             <{sub_type}> <{pred}> "" .
                         }}
                     """
                elif obj == "":
                    # If the object is an empty string
                    query = f"""
                             PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                             PREFIX owl: <http://www.w3.org/2002/07/owl#>
                             ASK
                             WHERE {{
                                 <{sub_type}> <{pred}> "" .
                             }}
                         """
                else:
                    # If the object is a literal, construct an ASK query
                    escaped_obj_value = obj.n3()
                    # Otherwise, construct the ASK query with the escaped object value
                    query = f"""
                             PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                             PREFIX owl: <http://www.w3.org/2002/07/owl#>
                             ASK
                             WHERE {{
                                 <{sub_type}> <{pred}> {escaped_obj_value} .
                             }}
                         """
            else:
                query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?res
                WHERE {{      
                        <{sub_type}> <{pred}> ?res  .
                        ?res rdf:type <{obj}> .
                    }}
            """
        except sparql.SPARQLExceptions.QueryBadFormed as e:
            print(f"Error in SPARQL query: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


        results = self.rdf_graph.query(query)
        return results

    def process_paths_based_on_class_labels(self, paths):
        """
            Process generated paths based on class labels.

            Args:
                paths (dict): A dictionary containing paths and class labels.

            Returns:
                dict: A dictionary containing mutually exclusive paths for each class label.

        """
        unique_paths_by_class = {}
        for key, value in paths.items():
            label = value[1]  # Get the label (1.0)
            walks = list(set([tuple(w) for w in value[0]]))  # Convert inner lists to tuples

            # Check if the label is already in the unique_paths_by_class
            if label in unique_paths_by_class:
                # Append the walks to the existing list for that label
                unique_paths_by_class[label].extend(walks)
            else:
                # Create a new entry for the label and initialize it with the walks
                unique_paths_by_class[label] = walks

        return unique_paths_by_class

    def replace_uris_in_path(self, paths):
        modified_paths = {}

        for key, value in paths.items():
            modified_paths[key] = {
                'label': value['label'],
                'path': '-'.join(str(term).split('#')[-1] for term in value['path']) if value['path'] else None
            }

        return modified_paths

    def format_data_for_metrics(self, test_data=None, predicted_data=None):
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
                for predicate_id, inner_pred_dict in predicted_data.items():
                    if test_id == predicate_id:
                        predicted_label = inner_pred_dict['label']
                        break

                predicted_predictions.append(predicted_label)

        # Create a DataFrame
        df = pd.DataFrame({
            'testid': test_ids,
            'actual_prediction': actual_predictions,
            'predicted_prediction': predicted_predictions
        }
        )
        df.to_csv('sparql_prediction.csv')
        return df['actual_prediction'], df['predicted_prediction']

    def _create_random_walk_object(self, data, algorithm, num_walks, walk_depth):
        if algorithm is None:
            # Create an object of RandomWalkWithoutBias
            rw_object = RandomWalkWithoutBias(self.rdf_graph, data, num_walks=num_walks,
                                         depth=walk_depth
                                         )
        else:
            # Create an object of the specified algorithm
            rw_object = algorithm(self.rdf_graph, data, num_walks=num_walks,
                                  depth=walk_depth
                                  )
        return rw_object
