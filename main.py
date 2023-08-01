from utils.decision_tree import DecisionTreeClassifier
from utils.random_walk import RandomWalk
from data.rdf import MUTAGDataset, AIFBDataset
from utils.operations import find_disjoint_lists, remove_uri_from_dict, merge_dict

if __name__ == "__main__":
    # TODO: Get the dataset from config file
    # TODO: Pre-processing need to be handled properly
    rootpath = './dataset/AIFB/aifb-hetero/'
    training_path = 'trainingSet.tsv'
    test_path = 'testSet.tsv'
    rdf_data = AIFBDataset(rootpath, training_path, test_path)
    graph = rdf_data.load_rdf_graph()
    data_dict = rdf_data.read_training_data()
    rw = RandomWalk(graph, data_dict, num_walks=6, walk_length=3, labelled=True)
    rw.set_random_walk()
    walks = rw.get_random_walk()
    most_occurring_pattern = rw.get_most_occurring_pattern_for_random_walks(count=20)
    mutually_exclusive_pattern = find_disjoint_lists(most_occurring_pattern)
    removed_uri_rpr = remove_uri_from_dict(mutually_exclusive_pattern, length=2)
    dt = DecisionTreeClassifier()
    m = dt.fit(removed_uri_rpr)
    print('Decision tree:', dt)
    'To predict'
    data_test = rdf_data.read_testing_data()
    merged_dict = merge_dict(data_test)
    rw = RandomWalk(graph, merged_dict, num_walks=5, walk_length=4)
    rw.set_random_walk()
    walks_test = rw.get_random_walk()
    removed_uri_rpr_test = remove_uri_from_dict(walks_test, length=2)
    x = dt.predict(removed_uri_rpr_test)