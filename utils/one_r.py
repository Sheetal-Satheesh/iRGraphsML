from utils.one_r_binary import OneRClassifierBinary
from utils.ovr_classifier import OVRClassifier


class OneRClassifier:
    """
    OneRClassifier is a wrapper class for OneRClassifierBinary (for binary classification) and OVRClassifier
    (for one-vs-rest multi-class classification) in RDF path prediction.

    Args:
        rdf_graph (rdflib.Graph): An RDF graph containing the knowledge base.
        class_names (list): A list of unique class names in the dataset.
        criterion (str): The criterion to evaluate the quality of a split in the decision tree.
                        Options: 'gini' for Gini impurity, 'entropy' for information gain.

    Attributes:
        rdf_graph (rdflib.Graph): The RDF graph containing the knowledge base.
        class_names (list): A list of unique class names in the dataset.
        criterion (str): The criterion to evaluate the quality of a split in the decision tree.
        classifier: An instance of OneRClassifierBinary or OVRClassifier based on the number of classes.

    Methods:
        fit(train_data, algorithm=None, num_walks=4, walk_depth=4):
            Fit the classifier using the training data and specified random walk parameters.

        predict(test_data, algorithm=None, num_walks=4, walk_depth=4):
            Make predictions on test data using the trained classifier and specified random walk parameters.

        plot_decision_tree():
            Plot the decision tree if the classifier has been trained. Otherwise, print a message.

    """
    def __init__(self, rdf_graph, class_names, criterion='gini'):
        self.rdf_graph = rdf_graph
        self.class_names = class_names
        self.criterion = criterion
        self.classifier = None

    def fit(self, train_data, algorithm=None, num_walks=4, walk_depth=4):
        unique_classes = self.class_names

        if len(unique_classes) == 2:
            # Binary Classification
            self.classifier = OneRClassifierBinary(self.rdf_graph,
                                                   class_name=self.class_names, criterion=self.criterion)
        else:
            # OVR Classification
            self.classifier = OVRClassifier(self.rdf_graph, class_names=self.class_names, criterion=self.criterion)

        self.classifier.fit(train_data, algorithm, num_walks, walk_depth)

    def predict(self, test_data, algorithm=None, num_walks=4, walk_depth=4):
        predictions, actual_label, decision_path = self.classifier.predict(test_data, algorithm, num_walks, walk_depth)
        return predictions, actual_label, decision_path

    def plot_decision_tree(self):
        if self.classifier is not None:
            self.classifier.plot_decision_tree()
        else:
            print("Classifier not trained yet.")