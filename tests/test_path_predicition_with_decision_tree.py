import unittest
from utils.rdf_path_prediction_with_decision_tree import DecisionTreePredictPath
from rdflib import URIRef, Graph, Literal, RDF


class TestDecisionTreePredictPath(unittest.TestCase):
    def setUp(self):
        # Create an empty RDF graph
        g = Graph()

        # Define some URIs for individuals in the family
        john = URIRef("http://example.org/john")
        jane = URIRef("http://example.org/jane")
        jerry = URIRef("http://example.org/jerry")
        minni = URIRef("http://example.org/minni")

        # Define some properties
        isParentOf = URIRef("http://example.org/isParentOf")
        isSiblingOf = URIRef("http://example.org/isSiblingOf")
        hasName = URIRef("http://example.org/hasName")
        isNieceOf = URIRef("http://example.org/isNieceOf")
        hasAge = URIRef("http://example.org/hasAge")
        isAgeGroupOf = URIRef("http://example.org/isAgeGroupOf")
        adult = URIRef("http://example.org/adult")
        young = URIRef("http://example.org/young")

        # Add statements to the graph
        g.add((john, RDF.type, URIRef("http://example.org/Person")))
        g.add((jane, RDF.type, URIRef("http://example.org/Person")))
        g.add((jerry, RDF.type, URIRef("http://example.org/Person")))
        g.add((minni, RDF.type, URIRef("http://example.org/Person")))
        g.add((john, RDF.type, adult))
        g.add((jane, RDF.type, adult))
        g.add((jerry, RDF.type, adult))
        g.add((minni, RDF.type, young))

        # Add ages for the new individuals
        g.add((john, hasAge, Literal(35)))
        g.add((jane, hasAge, Literal(30)))
        g.add((jerry, hasAge, Literal(22)))
        g.add((minni, hasAge, Literal(6)))

        g.add((john, hasName, Literal("John")))
        g.add((jane, hasName, Literal("Jane")))
        g.add((jerry, hasName, Literal("Jerry")))
        g.add((minni, hasName, Literal("Minni")))

        # Add age group information
        g.add((john, isAgeGroupOf, adult))
        g.add((jane, isAgeGroupOf, adult))
        g.add((jerry, isAgeGroupOf, adult))
        g.add((minni, isAgeGroupOf, young))

        g.add((john, isParentOf, minni))
        g.add((jane, isParentOf, minni))
        g.add((john, isSiblingOf, jerry))
        g.add((jerry, isSiblingOf, john))
        g.add((minni, isNieceOf, jerry))

        self.graph = g
        self.num_walks = 1
        self.walk_depth = 3

        # Create a dictionary to represent the training data
        self.training_data = {
            1.0: {'http://example.org/john': 35.0, 'http://example.org/jerry': 22.0},
            0.0: {'http://example.org/minni': 6.0}
        }

        # Create a dictionary to represent the test data
        self.test_data = {
            1.0: {'http://example.org/jane': 30}
        }

    def test_fit_and_predict(self):
        # Initialize the DecisionTreePredictPath class
        tree_predictor = DecisionTreePredictPath(self.graph, self.training_data)

        # Fit the decision tree classifier
        tree_predictor.fit(self.num_walks, self.walk_depth)

        # Ensure that the classifier is trained
        self.assertIsNotNone(tree_predictor.clf)

        # Predict labels for the test data
        predictions, true_labels = tree_predictor.predict(self.num_walks, self.walk_depth, self.test_data)

        # Check if predictions are of the correct length
        self.assertEqual(len(predictions), len(self.test_data))