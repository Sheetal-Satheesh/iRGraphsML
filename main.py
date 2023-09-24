from data.rdf import MUTAGDataset, AIFBDataset, DBLPDataset, BGSDataset
from utils.rdf_path_prediction_with_sparql import RDFPathClassifierWithSPARQl
from utils.rdf_path_prediction_with_decision_tree import DecisionTreePredictPath
from sklearn.metrics import f1_score, accuracy_score
from utils.operations import format_data_for_metrics


if __name__ == "__main__":
    rootpath = './raw_data/Mutag/mutag-hetero/'
    training_path = 'trainingSet.tsv'
    test_path = 'testSet.tsv'
    rdf_data = MUTAGDataset(rootpath, training_path, test_path)
    graph = rdf_data.load_rdf_graph()
    data_dict = rdf_data.read_training_data()
    prominent_class = rdf_data.get_most_prominent_class(data_dict)
    """
        Approach DECISION TREE
    """
    gp = DecisionTreePredictPath(graph, data_dict)
    gp.fit(4, 3)
    test_data = rdf_data.read_testing_data()
    predictions, actual_labels = gp.predict(5, 3, test_data)
    print('predictions', type(predictions))
    print('actual labels', actual_labels)

    # # Compute the F1 score
    f1 = f1_score(actual_labels, predictions)
    # Compute the accuracy
    accuracy = accuracy_score(actual_labels, predictions)
    # Display the F1 score and accuracy
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    """
        Approach: RANDOM WALK and Prediction With SPARQL
    """

    rw = RDFPathClassifierWithSPARQl(graph, prominent_class=prominent_class)
    path = rw.fit(data_dict, 4, 3)
    test_data = rdf_data.read_testing_data()
    predict = rw.predict(test_data)
    print(predict)
    df = format_data_for_metrics(test_data, predict)
    print(df.dtypes)
    print(type(df['actual_prediction']))
    # Compute the F1 score
    f1 = f1_score(df['actual_prediction'], df['predicted_prediction'])

    # Compute the accuracy
    accuracy = accuracy_score(df['actual_prediction'], df['predicted_prediction'])
    # Display the F1 score and accuracy
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
