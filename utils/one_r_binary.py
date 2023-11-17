from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from utils.baseclass_one_r import BaseClassifierOneR
import numpy as np


class OneRClassifierBinary(BaseClassifierOneR):
    """
       OneRClassifierBinary is a class for binary classification using the One Rule (OneR) algorithm in
       RDF KG.

       Args:
           rdf_graph (rdflib.Graph): An RDF graph containing the knowledge base.
           class_name (list): A list of unique class names in the dataset.
           criterion (str): The criterion to evaluate the quality of a split in the decision tree.
                           Options: 'gini' for Gini impurity, 'entropy' for information gain.

       Attributes:
           rdf_graph (rdflib.Graph): The RDF graph containing the knowledge base.
           class_name (list): A list of unique class names in the dataset.
           criterion (str): The criterion to evaluate the quality of a split in the decision tree.
           classifier: An instance of DecisionTreeClassifier for binary classification.

       Methods:
           fit(train_data, algorithm=None, num_walks=4, walk_depth=4):
               Fit the classifier using the training data and specified random walk parameters.

           predict(test_data, algorithm=None, num_walks=4, walk_depth=4):
               Make predictions on test data using the trained classifier and specified random walk parameters.

           plot_decision_tree():
               Plot the decision tree if the classifier has been trained. Otherwise, print a message.

       """
    def __init__(self, rdf_graph, class_name=None, criterion='gini'):
        # Call the constructor of the parent class (OneRClassifier)
        super().__init__(rdf_graph=rdf_graph, class_names=class_name, criterion=criterion)
        self.rules_ = None
        self.max_depth = 1
        self.feature_names_ = None
        self.class_name = class_name
        self.criterion = criterion
        self._estimator_type = 'classifier'
        self.rdf_graph = rdf_graph
        self.feature = None
        self.clf = None

    def __str__(self):
        '''
            Print out the list in a nice way
        '''

        s = '> ------------------------------\n> One R Rule List\n> ------------------------------\n'

        for rule in self.rules_:
            if 'col' in rule:
                prefix = f"if ~ {rule['col']} then class ==> {rule['class_name']}"
                val = f"if {rule['col']} then class ==> {rule['class_right']}"
                s += f"\t{prefix}\n\t{val}\n"
        return s

    def fit(self, train_data, algorithm, num_walks, walk_depth):
        super().fit(train_data, algorithm, num_walks, walk_depth
                    )
        y = self.y_train
        X = self.X_train
        self.rules_ = []

        m = DecisionTreeClassifier(max_depth=1, criterion=self.criterion)
        m.fit(X, y)
        col = m.tree_.feature[0]  # Get the feature at the self.feature_names_oot node
        cutoff = m.tree_.threshold[0]

        self.clf = m
        if col == -2:
            return []

        # y_left = y[X[:, col] < cutoff]
        # y_right = y[X[:, col] >= cutoff]

        # Access the class label for a specific leaf node (e.g., left and right leaf nodes)
        tree = self.clf.tree_

        root_node_belong_to_class_max_count = np.argmax(tree.value[0])
        left_leaf_node_val = np.argmax(tree.value[tree.children_left[0]])
        right_leaf_node_val = np.argmax(tree.value[tree.children_right[0]])
        root_class = root_node_belong_to_class_max_count

        par_node = {
            'col': self.feature_names_[col],
            'class_name': root_class,
            'index_col': col,
            'cutoff': cutoff,
            'class_right': right_leaf_node_val,
            'classifier': m  # Include the classifier key
        }
        self.rules_.append(par_node)
        return par_node

    def predict(self, test_data, algorithm, num_walks=4, walk_depth=4):
        super().predict(test_data, algorithm, num_walks, walk_depth)
        model_str = str(self)
        return model_str


    def plot_decision_tree(self):
        """
        Plot the trained decision trees for each class.

        """
        # Check if the classifier has been trained
        if len(self.rules_) == 0:
            print("Classifier has not been trained yet.")
            return

        # Plot the decision tree
        plt.figure(figsize=(16, 10))
        plot_tree(self.clf, filled=True, feature_names=self.feature_names_,
                  class_names=[str(label) for label in self.clf.classes_]
                  )
        plt.savefig('1R.pdf')

