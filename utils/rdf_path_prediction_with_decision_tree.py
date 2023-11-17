from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias
from utils.operations import remove_uri_from_dict
import numpy as np


class DecisionTreePredictPath:
    def __init__(self, rdf_graph):
        """
           Initialize the DecisionTreePredictPath class.

           Args:
               rdf_rdf_graph: The rdf_graph representing the data.
               train_data: A dictionary containing the Training Data and corresponding labels.

        """
        self.rdf_graph = rdf_graph
        self.train_data = None
        self.clf = None
        self.feature = None

    def __str__(self):
        """
            Return a string representation of the trained decision tree.

            Returns:
                str: A string representation of the decision tree.

        """
        if self.clf is None:
            return "Classifier has not been trained yet."

        # Get class labels (replace with your actual class labels)
        class_labels = self.feature['label'].unique().tolist()  # Add names for all classes

        # Initialize the string representation
        tree_str = ""

        # Create a string representation of the decision tree
        tree_str += self._tree_to_str(self.clf.tree_, self.feature.columns.drop(['label', 'instance']), class_labels)

        return tree_str

    def _tree_to_str(self, tree, feature_names, class_labels, node_index=0):
        """
            Convert the decision tree to a string representation recursively.

            Args:
                tree: The decision tree.
                feature_names: List of feature names.
                class_labels: List of class labels.
                node_index: The index of the current node.

            Returns:
                str: A string representation of the decision tree.

        """
        indent = " "
        # Recursive function to convert the decision tree to a string
        if tree.feature[node_index] == -2:
            # Leaf node
            class_index = np.argmax(tree.value[node_index])
            return f"{indent}return {class_labels[class_index]}\n"
        else:
            # Non-leaf node
            feature_index = tree.feature[node_index]
            feature_name = feature_names[feature_index]
            threshold = tree.threshold[node_index]

            left_child = self._tree_to_str(tree, feature_names, class_labels, tree.children_left[node_index],)
            right_child = self._tree_to_str(tree, feature_names, class_labels, tree.children_right[node_index],)

            if node_index == 0:
                condition = f"if {feature_name} <= {threshold:.2f}:\n"
            else:
                condition = f"elif {feature_name} > {threshold:.2f}:\n"

            return f"{indent}{condition}{left_child}{indent}else:\n{right_child}"

    def fit(self, train_data,  num_walks, walk_depth):
        """
            Fit the decision tree classifier to the training data.

            Args:
                num_walks: Number of random walks to perform.
                walk_depth: Depth of the random walks.
                :param train_data:  Input training data

        """
        self.train_data = train_data  # Store data_dict for later use
        rw1 = BiasedRandomWalk(self.rdf_graph, self.train_data, num_walks=num_walks, depth=walk_depth
                         )
        rw1.set_random_walk()
        walks = rw1.get_random_walk()
        removed_uri = remove_uri_from_dict(walks)
        df = self._convert_paths_into_features(removed_uri)

        X = df.drop(['label', 'instance'], axis=1)

        y = df['label']

        # Create a DecisionTreeClassifier with the Gini criterion
        clf = DecisionTreeClassifier(criterion='gini', random_state=3, max_depth=6, ccp_alpha=0.01, min_samples_leaf=2)

        # Fit the classifier to the data
        clf.fit(X, y)

        self.feature = df
        self.clf = clf

    def predict(self, test_data, num_walks, walk_depth):
        """
            Predict labels for test data using the trained classifier.

            Args:
                num_walks: Number of random walks to perform for test data.
                walk_depth: Depth of the random walks for test data.
                test_data: Test data for prediction.

            Returns:
                tuple: A tuple containing predicted labels and true labels.

        """
        rw1 = BiasedRandomWalk(self.rdf_graph, test_data, num_walks=num_walks, depth=walk_depth
                         )
        rw1.set_random_walk()
        walks = rw1.get_random_walk()
        removed_uri = remove_uri_from_dict(walks)
        test_df = self._preprocess_test_data(removed_uri)
        X = test_df.drop(['label', 'instance'], axis=1)
        # Predict class labels
        predictions = self.clf.predict(X)

        # Predict class probabilities (scores)
        prediction_scores = self.clf.predict_proba(X)

        # Get the decision path for each prediction
        decision_paths = self.get_decision_paths(X)

        # Create a list of results, each containing test_id, path, label, and scores
        results = []
        for i in range(len(X)):
            result = {
                "test_id": test_df['instance'].iloc[i],  # Replace with the actual patient_id source
                "path": decision_paths[i],
                "label": predictions[i],
                "scores": prediction_scores[i]  # This contains the class probabilities
            }
            results.append(result)
        print(f'Results,{results}')

        return predictions, test_df['label']

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
        df.to_csv('ttttt.tsv')
        self.feature.to_csv('train.csv')
        return df

    def get_decision_paths(self, X):
        """
            Get decision paths for each sample in the test data.

            Args:
                X: Test data for which decision paths are calculated.

            Returns:
                list: List of decision paths for each sample.

        """
        # Initialize a list to store decision paths
        decision_paths = []
        feature_names = X.columns.tolist()
        print(feature_names)

        # Access the underlying tree structure
        tree = self.clf.tree_

        for index, row in X.iterrows():
            # Initialize the decision path for this sample
            sample_decision_path = []

            # Start at the root node of the tree
            node_index = 0  # Root node index

            while True:
                # Extract the feature and threshold at this node
                feature_index = tree.feature[node_index]
                threshold = tree.threshold[node_index]

                if feature_index == -2:
                    # Handle leaf node: Append the prediction value and break
                    prediction = tree.value[node_index]
                    sample_decision_path.append(("Leaf", prediction))
                    break

                # Append the feature name and threshold to the decision path
                feature_name = feature_names[feature_index]
                sample_decision_path.append(feature_name)

                if row[feature_name] <= threshold:
                    # Follow the left child
                    child_node_index = tree.children_left[node_index]
                else:
                    # Follow the right child
                    child_node_index = tree.children_right[node_index]

                # Move to the child node
                node_index = child_node_index

            # Append the decision path for this sample to the list
            decision_paths.append(sample_decision_path)

        return decision_paths

    def plot_decision_tree(self):
        """
            Plot the trained decision tree.

        """
        # Check if the classifier has been trained
        if self.clf is None:
            print("Classifier has not been trained yet.")
            return

        # Plot the decision tree
        plt.figure(figsize=(16, 10))
        plot_tree(self.clf, filled=True, feature_names=self.feature.columns.drop(['label', 'instance']),
                  class_names=[str(label) for label in self.clf.classes_])
        plt.savefig('Decision_tree_1.pdf')

    def calculate_feature_importance(self):
        """
                    Calculate and return feature importances from the trained decision tree.

                    Returns:
                        dict: A dictionary with feature names as keys and their importances as values.
                """
        if self.clf is None:
            print("Classifier has not been trained yet.")
            return None

        feature_importances = dict(
            zip(self.feature.columns.drop(['label', 'instance']), self.clf.feature_importances_)
            )
        return feature_importances
