from utils.decision_tree import DecisionTreeClassifier
from utils.random_walk import RandomWalk
from data.rdf import MUTAGDataset, AIFBDataset
from utils.operations import find_disjoint_lists, remove_uri_from_dict, merge_dict
from utils.metric import EvaluationMetrics as em
from owlrl import convert_graph
from utils.sparql import DecodeRWSparQL
import rdflib
import copy


if __name__ == "__main__":
    # TODO: Get the dataset from config file
    # TODO: Pre-processing need to be handled properly
    # TODO: Evaluation Metric
    rootpath = './dataset/Mutag/mutag-hetero/'
    training_path = 'trainingSet.tsv'
    test_path = 'testSet.tsv'
    rdf_data = MUTAGDataset(rootpath, training_path, test_path)
    graph = rdf_data.load_rdf_graph()
    data_dict = rdf_data.read_training_data()
    print(data_dict)
    rw = RandomWalk(graph, data_dict, num_walks=5, depth=3, labelled=True)
    rw.set_random_walk()
    walks = rw.get_random_walk()
    most_occurring_pattern = rw.get_most_occurring_pattern_for_random_walks(count=10)
    new_mst_occurring_pattern = copy.deepcopy(most_occurring_pattern)
    mutually_exclusive_pattern = find_disjoint_lists(new_mst_occurring_pattern)
    removed_uri_rpr = remove_uri_from_dict(mutually_exclusive_pattern, length=2)
    """
      Approach: RANDOM WALK and CHECK THE WALK
      """
    test_data = rdf_data.read_testing_data()
    srw = DecodeRWSparQL(mutually_exclusive_pattern, graph)
    result = srw.predict(test_data)
    print(result)
    """
    Approach DECISION TREE
    """
    #
    # dt = DecisionTreeClassifier()
    # m = dt.fit(removed_uri_rpr)
    # 'To predict'
    # data_test = rdf_data.read_testing_data()
    # merged_dict = merge_dict(data_test)
    # rw = RandomWalk(graph, merged_dict, num_walks=5, depth=3)
    # rw.set_random_walk()
    # walks_test = rw.get_random_walk()
    # removed_uri_rpr_test = remove_uri_from_dict(walks_test, length=1)
    # x = dt.predict(removed_uri_rpr_test)
    # print(x)

