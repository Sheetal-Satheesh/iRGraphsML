import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from utils.baseclass_one_r import BaseClassifierOneR

class OVRClassifier(BaseClassifierOneR):
    """
       OVRClassifier is a class for one-vs-rest multi-class classification using One Rule (OneR) algorithm
       with decision trees in RDF path prediction.

       Args:
           rdf_graph (rdflib.Graph): An RDF graph containing the knowledge base.
           class_names (list): A list of unique class names in the dataset.
           criterion (str): The criterion to evaluate the quality of a split in the decision tree.
                           Options: 'gini' for Gini impurity, 'entropy' for information gain.

       Attributes:
           rdf_graph (rdflib.Graph): The RDF graph containing the knowledge base.
           class_names (list): A list of unique class names in the dataset.
           criterion (str): The criterion to evaluate the quality of a split in the decision tree.
           clf: An instance of OneVsRestClassifier with DecisionTreeClassifier for one-vs-rest classification.
           rules_ (list): A list containing rule information for each class.

       Methods:
           __init__(rdf_graph, class_names=None, criterion='gini'):
               Initialize the OVRClassifier.

           __str__():
               Return a string representation of the OVRClassifier, displaying the One Rule (OneR) classifier rules.

           fit(train_data, algorithm=None, num_walks=4, walk_depth=4):
               Fit the classifier using the training data and specified random walk parameters.

           predict(test_data, algorithm=None, num_walks=4, walk_depth=4):
               Make predictions on test data using the trained classifier and specified random walk parameters.

           plot_decision_trees_ovr():
               Plot the decision trees for each class in one-vs-rest classification.

       """
    def __init__(self, rdf_graph, class_names=None, criterion='gini'):
        super().__init__(rdf_graph, class_names, criterion)
        self.clf = None

    def __str__(self):
        '''
                    Print out the list in a nice way
        '''

        s = '> ------------------------------\n> OvR Classifier Rule List\n> ------------------------------\n'

        if self.rules_ is None:
            return "--No rules available--"

        for rule in self.rules_:
            if 'col' in rule:
                class_name = str(rule['class_name'])  # Convert to string
                class_right = str(rule['class_right'])  # Convert to string

                prefix = f"if ~ {rule['col']} then class ==> {class_name}"
                val = f"if {rule['col']} then class ==> {class_right}"

                # Check if the predicted class is not "0" before printing
                if class_name != "0":
                    s += f"\t{prefix}\n"
                # Check if the predicted class_right is not "0" before printing
                if class_right != "0":
                    s += f"\t{val}\n"

        return s

    def fit(self, train_data, algorithm=None, num_walks=4, walk_depth=4):
        super().fit(train_data, algorithm, num_walks, walk_depth)
        y = self.y_train
        X = self.X_train
        if not self.class_names:
            self.class_names = np.unique(y)
            self.class_counts = [self.count_instances_for_class(train_data, cn) for cn in self.class_names]
            print(self.class_counts)
        self.clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=1)).fit(X, y)
        self._extract_rule()

    def predict(self, test_data, algorithm=None, num_walks=4, walk_depth=4):
        super().predict(test_data, algorithm, num_walks, walk_depth)
        model_str = str(self)
        return model_str

    def plot_decision_trees_ovr(self):
        class_names = self.class_names
        classifier = self.clf
        feature_names = self.feature_names_

        for i, (class_name, estimator) in enumerate(zip(class_names, classifier.estimators_)):
            plot_tree(estimator, filled=True, feature_names=feature_names,
                      class_names=[f"Not {class_name}", class_name]
                      )
            plt.savefig(f'{class_name}_ovr.pdf')


    def _extract_rule(self):
        class_names = self.class_names
        classifier = self.clf
        self.rules_ = []

        for i, (class_name, estimator) in enumerate(zip(class_names, classifier.estimators_)):
            col = estimator.tree_.feature[0]  # Get the feature at the root node
            cutoff = estimator.tree_.threshold[0]


            if col == -2:
                return []

            # Access the class label for a specific leaf node (e.g., left and right leaf nodes)
            tree = estimator.tree_
            left_leaf_node_val = np.argmax(tree.value[tree.children_left[0]])

            par_node = {
                'col': self.feature_names_[col],
                'class_name': class_name,
                'index_col': col,
                'cutoff': cutoff,
                'class_right': left_leaf_node_val,
                'classifier': estimator  # Include the classifier key
            }
            self.rules_.append(par_node)