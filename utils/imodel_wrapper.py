from pandas import DataFrame
import pandas as pd
from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias
from utils.operations import remove_uri_from_dict
from imodels import HSTreeClassifier, FIGSClassifier, BoostedRulesClassifier
import numpy as np

class BaseWrapper:
    """Base class for wrapper functionality."""
    
    def __init__(self, rdf_graph):
        self.test_df = None
        self.test_data = None
        self.rdf_graph = rdf_graph
        self.feature = None
        self.feature_names_ = None
        self.train_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None

    def process_fit_data(self, train_data, algorithm, num_walks, walk_depth):
        """
               Process training data and create features for training.

               Parameters:
               - train_data: Training data.
               - algorithm: Random walk algorithm.
               - num_walks: Number of walks.
               - walk_depth: Walk depth.
        """
        self.train_data = train_data  # Store data_dict for later use
        rw1 = self._create_random_walk_object(train_data, algorithm, num_walks, walk_depth)
        rw1.set_random_walk()
        walks = rw1.get_random_walk()
        removed_uri = remove_uri_from_dict(walks)
        df = self._convert_paths_into_features(removed_uri)
        X = df.drop(['label', 'instance'], axis=1)
        self.feature = df
        self.X_train = X.to_numpy()
        y = df['label']
        self.y_train = y.to_numpy()
        self.feature_names_ = df.columns.difference(['label', 'instance']).tolist()

    def process_predict_data(self, test_data, algorithm, num_walks=4, walk_depth=4):
        """
                Process test data and create features for testing.

                Parameters:
                - test_data: Test data.
                - algorithm: Random walk algorithm.
                - num_walks: Number of walks.
                - walk_depth: Walk depth.
        """
        self.test_data = test_data
        rw1 = self._create_random_walk_object(test_data, algorithm, num_walks, walk_depth)
        rw1.set_random_walk()
        walks = rw1.get_random_walk()
        removed_uri = remove_uri_from_dict(walks)
        test_df: DataFrame = self._preprocess_test_data(removed_uri)
        self.test_df = test_df
        X_test = test_df.drop(['label', 'instance'], axis=1)
        self.X_test = X_test.to_numpy()

    def _preprocess_test_data(self, path_sequences):
        """
            Preprocess test data to match the format of training data.

            Args:
                path_sequences: Dictionary containing path sequences for test data.

            Returns:
                pd.DataFrame: Preprocessed test data.
        """
        # Create an empty DataFrame with columns from the training data
        df = pd.DataFrame(columns=self.feature.columns)

        # Create a list of dictionaries to hold the data
        data_list = []

        # Iterate through path sequences and update values if they exist in test_data
        for key, value in path_sequences.items():
            instance = key
            labels = ['->'.join(nested_list) for nested_list in value[0]]
            label_value = value[1]

            # Create a dictionary for each entry
            entry = {'instance': instance, 'label': label_value}

            # Check if at least one label is in df.columns
            labels_in_df = [label for label in labels if label in df.columns]

            if labels_in_df:
                # Update values if the path sequence exists in test_data
                for label in labels_in_df:
                    entry[label] = 1
                data_list.append(entry)
            else:
                data_list.append(entry)

        # Create the DataFrame
        df = pd.DataFrame(data_list)

        # Add columns that are in the training data but not in the test data
        missing_columns = list(set(self.feature.columns) - set(df.columns))
        # for col in missing_columns:
        #     df[col] = 0
        if missing_columns:
            missing_columns_df = pd.DataFrame(0, index=df.index, columns=missing_columns)
            df = pd.concat([df, missing_columns_df], axis=1)

        df = df.fillna(0)
        # Reorder the columns to match the order in the training data
        df = df[self.feature.columns]
        df.to_csv('Trained Feature Table.tsv')
        self.feature.to_csv('Trained Feature Table.csv')
        return df

    def _convert_paths_into_features(self, path_sequences):
        """
            Convert path sequences into feature vectors for training data.

            Args:
                path_sequences: Dictionary containing path sequences.

            Returns:
                pd.DataFrame: A DataFrame with features for training.

        """
        # Convert the dictionary into a list of dictionaries
        data_list = []
        for key, value in path_sequences.items():
            instance = key
            labels = ['->'.join(nested_list) for nested_list in value[0]]
            label_value = value[1]

            # Create a dictionary for each entry
            entry = {'instance': instance, 'label': label_value}
            for label in labels:
                entry[label] = 1
            data_list.append(entry)

        df = pd.DataFrame(data_list)
        df = df.fillna(0)
        df.to_csv('train.tsv')
        return df

    def count_instances_for_class(self, data_dict, class_name):
        """
        Count the instances for a given class name in the data dictionary.

        Args:
            data_dict (dict): A dictionary containing class labels as keys,
                             and their corresponding values are dictionaries.
            class_name (str): The class name for which you want to count instances.

        Returns:
            int: The count of instances for the specified class name.
        """
        if class_name in data_dict:
            return len(data_dict[class_name])
        else:
            return 0

    def _create_random_walk_object(self, data, algorithm, num_walks, walk_depth):
        """
                Create a random walk object based on the specified algorithm.

                Parameters:
                - data: Data for random walk.
                - algorithm: Random walk algorithm.
                - num_walks: Number of walks.
                - walk_depth: Walk depth.

                Returns:
                - Random walk object.
        """
        if algorithm is None:
            # Create an object of BiasedRandomWalk
            rw_object = BiasedRandomWalk(self.rdf_graph, data, num_walks=num_walks,
                                         depth=walk_depth
                                         )
        else:
            # Create an object of the specified algorithm
            rw_object = algorithm(self.rdf_graph, data, num_walks=num_walks,
                                  depth=walk_depth
                                  )
        return rw_object


class imodelsWrapperFigs(BaseWrapper, FIGSClassifier):
    """Wrapper for FIGSClassifier built around imodels FIGSClassifier."""

    def __init__(self, rdf_graph, max_rules: int = 5, max_trees: int = None,
                 min_impurity_decrease: float = 0.0, random_state=None, max_features: str = None):
        """
                Initialize imodelsWrapperFigs.

                Parameters:
                - rdf_graph: RDF graph.
                - max_rules: Maximum number of rules. In imodels it is set to 12 however, we set it to 5
                - max_trees: Maximum number of trees.
                - min_impurity_decrease: Minimum impurity decrease.
                - random_state: Random state.
                - max_features: Maximum features.
        """
        BaseWrapper.__init__(self, rdf_graph)
        FIGSClassifier.__init__(self, max_rules, max_trees, min_impurity_decrease, random_state, max_features)
        self.flag_multi_class = False
        self.label_mapping = None

    def fit(self, train_data, algorithm, num_walks, walk_depth, feature_names=None,
            verbose=False, sample_weight=None, categorical_features=None):
        super().process_fit_data(train_data, algorithm, num_walks, walk_depth)
        self.feature_names = self.feature_names_
        labels = self.y_train
        print('Labels', labels)

        # Check if the task is binary
        is_binary_classification = len(np.unique(labels))
        if is_binary_classification != 2:
            self.flag_multi_class = True
            if np.issubdtype(labels.dtype, np.str_) or np.issubdtype(labels.dtype, np.object_):
                unique_labels, integer_labels = np.unique(labels, return_inverse=True)
                self.y_train = integer_labels
                self.label_mapping = dict(zip(range(len(unique_labels)), unique_labels))

        super().fit(self.X_train, self.y_train, self.feature_names, verbose,
                    sample_weight, categorical_features
                    )

    def predict(self, test_data, algorithm, num_walks=4, walk_depth=4, categorical_features=None):
        super().process_predict_data(test_data, algorithm, num_walks, walk_depth)
        pred = super().predict(self.X_test, categorical_features)

        if self.flag_multi_class:
            predicted_labels = np.vectorize(self.label_mapping.get)(pred)
            return self.test_df['label'], predicted_labels

        return self.test_df['label'], pred

    def predict_proba(self, test_data, algorithm, num_walks=4, walk_depth=4,
                      categorical_features=None, use_clipped_prediction=False):
        super().process_predict_data(test_data, algorithm, num_walks, walk_depth)
        pred_proba = super().predict_proba(self.X_test, categorical_features, use_clipped_prediction)

        if self.flag_multi_class:
            predicted_labels = np.vectorize(self.label_mapping.get)(pred_proba)
            return self.test_df['label'], predicted_labels

        return pred_proba


class imodelsWrapperHS(BaseWrapper, HSTreeClassifier):
    """Wrapper for HSTreeClassifier built around imodels HSTreeClassifier."""

    def __init__(self, rdf_graph, max_leaf_node=5):
        """
               Initialize imodelsWrapperHS.

               Parameters:
               - rdf_graph: RDF graph.
               - max_leaf_node: Maximum number of leaf nodes. We set the max leaf nodes to 5.
        """
        self.flag_multi_class = False
        self.label_mapping = None
        self.max_leaf_node = max_leaf_node
        BaseWrapper.__init__(self, rdf_graph)
        HSTreeClassifier.__init__(self, max_leaf_nodes=self.max_leaf_node)
        self.feature_names = None

    def fit(self, train_data, algorithm, num_walks, walk_depth):
        super().process_fit_data(train_data, algorithm, num_walks, walk_depth)
        self.feature_names = self.feature_names_
        labels = self.y_train

        # Check if the task is binary
        is_binary_classification = len(np.unique(labels))
        if is_binary_classification != 2:
            self.flag_multi_class = True
            if np.issubdtype(labels.dtype, np.str_) or np.issubdtype(labels.dtype, np.object_):
                unique_labels, integer_labels = np.unique(labels, return_inverse=True)
                self.y_train = integer_labels
                self.label_mapping = dict(zip(range(len(unique_labels)), unique_labels))

        super().fit(self.X_train, self.y_train, feature_names=self.feature_names_)

    def predict(self, test_data, algorithm, num_walks=4, walk_depth=4):
        super().process_predict_data(test_data, algorithm, num_walks, walk_depth)
        pred = super().predict(self.X_test, self.feature_names)

        if self.flag_multi_class:
            predicted_labels = np.vectorize(self.label_mapping.get)(pred)
            return self.test_df['label'], predicted_labels

        return self.test_df['label'], pred

    def predict_proba(self, test_data, algorithm, num_walks=4, walk_depth=4):
        super().process_predict_data(test_data, algorithm, num_walks, walk_depth)
        pred_proba = super().predict_proba(self.X_test, self.feature_names)

        if self.flag_multi_class:
            predicted_labels = np.vectorize(self.label_mapping.get)(pred_proba)
            return self.test_df['label'], predicted_labels

        return pred_proba
