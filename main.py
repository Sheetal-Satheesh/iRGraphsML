from utils.decision_tree import DecisionTreeClassifier
from utils.random_walk import IGraphMlRandomWalk
from data.rdf import MUTAGDataset, AIFBDataset
from utils.operations import find_disjoint_lists, remove_uri_from_dict

if __name__ == "__main__":
    # TODO: Get the dataset from config file
    rootpath = './dataset/Mutag/mutag-hetero/'
    training_path = 'trainingSet.tsv'
    test_path = 'testSet.tsv'
    rdf_data = MUTAGDataset(rootpath, training_path, test_path)
    graph = rdf_data.load_rdf_graph()
    print(graph)
    data_dict = rdf_data.read_training_data()
    print(data_dict)
    rw = IGraphMlRandomWalk(graph, data_dict, num_walks=5, walk_length=3)
    rw.set_random_walk()
    walks = rw.get_random_walk()
    most_occurring_pattern = rw.get_most_occurring_pattern_for_random_walks(count=20)
    print(most_occurring_pattern)

    mutually_exclusive_pattern = find_disjoint_lists(most_occurring_pattern)
    print(mutually_exclusive_pattern)
    removed_uri_rpr = remove_uri_from_dict(mutually_exclusive_pattern, length=2)
    print(f'removed uri{removed_uri_rpr}')
    dt = DecisionTreeClassifier()
    m = dt.fit(removed_uri_rpr)
    print(dt)
