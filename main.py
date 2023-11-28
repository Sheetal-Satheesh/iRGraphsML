from data.rdf import MUTAGDataset, AIFBDataset, BGSDataset, CarcinogenesisDataset
from utils.rdf_sparql_node_classifier import RDFNodeClassifierWithSPARQL
from utils.rdf_decision_tree_node_classifier import RDFDecisionTreeNodeClassifier
from sklearn.metrics import f1_score, accuracy_score
from utils.one_r import OneRClassifier
from utils.random_walk import BiasedRandomWalk, RandomWalkWithoutBias


if __name__ == "__main__":
    rootpath = './raw_data/AIFB/aifb-hetero/'
    training_path = 'trainingSet.tsv'
    test_path = 'testSet.tsv'
    rdf_data = AIFBDataset(rootpath, training_path, test_path)
    graph = rdf_data.load_rdf_graph()
    class_names = rdf_data.get_unique_classes()
    data_dict = rdf_data.read_training_data()
    test_data = rdf_data.read_testing_data()
    prominent_class = rdf_data.get_most_prominent_class(test_data)

    """
        Approach DECISION TREE
    """
    gp = RDFDecisionTreeNodeClassifier(graph)
    gp.fit(data_dict, None, 4, 3)
    test_data = rdf_data.read_testing_data()
    predictions, actual_labels = gp.predict(test_data, None, 4, 3)
    print('predictions', type(predictions))
    print('actual labels', actual_labels)

    # # Compute the F1 score
    f1 = f1_score(actual_labels, predictions)
    # Compute the accuracy
    accuracy = accuracy_score(actual_labels, predictions)
    # Display the F1 score and accuracy
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(gp)
    gp.plot_decision_tree()
    """
            Approach OneR
    """

    clf = OneRClassifier(graph, class_names)
    clf.fit(data_dict, BiasedRandomWalk,  4, 3)
    prediction, actual, _ = clf.predict(test_data, None, 4, 3)
    print(prediction)
    f1 = f1_score(actual, prediction, average='macro')
    # Display the F1 score and accuracy
    accuracy = accuracy_score(actual, prediction)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    #
    """
        Approach: RANDOM WALK and Prediction With SPARQL
    """

    rw = RDFNodeClassifierWithSPARQL(graph, algorithm=RandomWalkWithoutBias, prominent_class=prominent_class)
    path = rw.fit(data_dict, 4, 3)
    test_data = rdf_data.read_testing_data()
    predictions, actual_labels = rw.predict(test_data)
    print(rw)
    f1 = f1_score(actual_labels, predictions, average='macro')
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
