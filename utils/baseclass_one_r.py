from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias
from utils.operations import remove_uri_from_dict
from abc import ABC


class BaseClassifierOneR(ABC):
    """
        BaseClassifierOneR is an abstract base class for implementing One Rule (OneR) classifiers with decision trees
        in RDF path prediction.

        Args:
            rdf_graph (rdflib.Graph): An RDF graph containing the knowledge base.
            class_names (list): A list of unique class names in the dataset.
            criterion (str): The criterion to evaluate the quality of a split in the decision tree.
                            Options: 'gini' for Gini impurity, 'entropy' for information gain.

        Attributes:
            rdf_graph (rdflib.Graph): The RDF graph containing the knowledge base.
            test_data (None or dict): The test data for making predictions.
            feature (None or pd.DataFrame): The feature DataFrame for training data.
            feature_names_ (None or list): The list of feature names.
            train_data (None or dict): The training data for fitting the classifier.
            X_train (None or np.ndarray): The feature matrix for training data.
            y_train (None or np.ndarray): The target labels for training data.
            X_test (None or np.ndarray): The feature matrix for test data.
            class_names (list): A list of unique class names in the dataset.
            criterion (str): The criterion to evaluate the quality of a split in the decision tree.
            classifier (DecisionTreeClassifier): The decision tree classifier with max_depth=1.
            rules_ (None or list): The list containing rule information for each class.
            class_counts (None or list): The list containing counts of instances for each class.

        Methods:
            __init__(rdf_graph, class_names, criterion='gini'):
                Initialize the BaseClassifierOneR.

            __str__():
                Return a string representation of the classifier.

            fit(train_data, algorithm, num_walks=4, walk_depth=4):
                Fit the classifier using the training data and specified random walk parameters.

            predict(test_data, algorithm, num_walks=4, walk_depth=4):
                Make predictions on test data using the trained classifier and specified random walk parameters.

            _preprocess_test_data(path_sequences):
                Preprocess test data to match the format of training data.

            _convert_paths_into_features(path_sequences):
                Convert path sequences into feature vectors for training data.

            count_instances_for_class(data_dict, class_name):
                Count the instances for a given class name in the data dictionary.

            _create_random_walk_object(data, algorithm, num_walks, walk_depth):
                Create a random walk object based on the specified algorithm.

        """
    def __init__(self, rdf_graph, class_names, criterion='gini'):
        self.test_data = None
        self.rdf_graph = rdf_graph
        self.feature = None
        self.feature_names_ = None
        self.train_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.class_names = class_names
        self.criterion = criterion
        self.classifier = DecisionTreeClassifier(max_depth=1, criterion=criterion)
        self.rules_ = None
        self.class_counts = None

    def __str__(self):
        pass

    def fit(self, train_data, algorithm, num_walks, walk_depth):
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

    def predict(self, test_data, algorithm, num_walks=4, walk_depth=4):
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
            labels = ['-'.join(nested_list) for nested_list in value[0]]
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
        missing_columns = set(self.feature.columns) - set(df.columns)
        for col in missing_columns:
            df[col] = 0

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
            labels = ['-'.join(nested_list) for nested_list in value[0]]
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
