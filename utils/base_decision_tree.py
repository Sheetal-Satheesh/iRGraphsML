from abc import ABC, abstractmethod
import pandas as pd
from utils.operations import remove_uri_from_dict
from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias


class BaseDecisionTree(ABC):
    def __init__(self, rdf_graph):
        """
           Initialize the BaseDecisionTreePredictPath class.

           Args:
               rdf_graph: The rdf_graph representing the data.

        """
        self.test_data = None
        self.rdf_graph = rdf_graph
        self.feature = None
        self.feature_names_ = None
        self.train_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.classifier = None
        self.rules_ = None
        self.class_counts = None
        self.test_df = None

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

    def predict(self, test_data, algorithm, num_walks, walk_depth):
        rw = self._create_random_walk_object(test_data, algorithm, num_walks, walk_depth)
        rw.set_random_walk()
        walks = rw.get_random_walk()
        removed_uri = remove_uri_from_dict(walks)
        self.test_df = self._preprocess_test_data(removed_uri)
        self.X_test = self.test_df.drop(['label', 'instance'], axis=1)

    @abstractmethod
    def _tree_to_str(self, tree, feature_names, class_labels, node_index=0):
        pass

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
        df.to_csv('Dataframe_after_processing.tsv')
        self.feature.to_csv('Features_after_training.csv')
        return df

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
        print(path_sequences)
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
