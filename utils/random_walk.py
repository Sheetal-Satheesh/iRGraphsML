from abc import ABCMeta
import random
from collections import Counter
from rdflib import Graph, URIRef, RDF, OWL


class RandomWalk(metaclass=ABCMeta):
    def __init__(self, graph, data_dict, num_walks=4, depth=3, labelled=False):
        self.random_walk = {}
        self.data_dict = data_dict
        self.graph = graph
        self.num_walks = num_walks
        self.depth = depth
        self.labelled = labelled
        self.most_occurring_pattern = {}

    def get_random_walk(self):
        return self.random_walk

    def set_random_walk(self):
        self._random_walk(self.data_dict, self.graph, self.num_walks,  self.depth, self.labelled)

    def get_most_occurring_pattern_for_random_walks(self, count=15):
        if bool(self.random_walk):
            unique_labels = self.random_walk.keys()
            for cls in unique_labels:
                pattern = self.check_most_occurring_pattern(self.random_walk[cls], count=count)
                if cls not in self.most_occurring_pattern:
                    self.most_occurring_pattern[cls] = pattern
            return self.most_occurring_pattern
        else:
            raise Exception('Random Walks must be performed to find the most frequent pattern.')

    def clean(self):
        self.random_walk = None

    def _random_walk(self, data_dict, graph, num_walks=2, depth=3, labelled=False):

        if labelled:
            classes_for_rw = data_dict.keys()
            for cls in classes_for_rw:
                training_nodes = data_dict[cls]
                path_sequences = self.generate_random_walks(training_nodes, graph, num_walks, depth)
                if cls not in self.random_walk:
                    self.random_walk[cls] = path_sequences
        else:
            nodes = data_dict.keys()
            for node in nodes:
                path_sequences = self.generate_random_walks([node], graph, num_walks, depth)
                if node not in self.random_walk:
                    self.random_walk[node] = path_sequences


    def get_classes_of_current_node(self, graph, current_node):
        # Get the classes of the current node
        class_triples = graph.triples((current_node, RDF.type, None))
        classes_of_current_node = []
        for _, _, class_ in class_triples:
            classes_of_current_node.append(class_)
        return classes_of_current_node

    def check_most_occurring_pattern(self, patterns, count=10):
        # Count the occurrences of each unique list
        counts = Counter(tuple(sublist) for sublist in patterns)

        # Find the list with the maximum count
        most_common_list = max(counts, key=counts.get)
        most_occuring_random_walk_occurrences = [item[0] for item in counts.most_common(count)]
        occurrence_count = counts[most_common_list]

        return most_occuring_random_walk_occurrences

    def generate_random_walks(self, nodes, graph, num_walks, depth):
        path_sequences = []
        for node in nodes:
            current_node = URIRef(node)
            classes_of_current_node = self.get_classes_of_current_node(graph, current_node)
            class_of_current_node = random.choice(classes_of_current_node)

            for walk in range(num_walks):
                path_sequence = [class_of_current_node]
                for w in range(depth - 1):
                    # Get all outgoing predicates from the current node
                    predicates = list(graph.predicates(subject=current_node))
                    predicates = [predicate for predicate in predicates if predicate != RDF.type]
                    # print(f'Predicates:{predicates}')
                    if not predicates:
                        break  # Stop the random walk if there are no outgoing predicates

                    # Choose a random edge/predicate
                    random_predicate = predicates[random.randint(0, len(predicates) - 1)]

                    # Get all objects connected by the random predicate
                    objects = list(graph.objects(subject=current_node, predicate=random_predicate))

                    if not objects:
                        break

                    # Choose a random object
                    random_object = objects[random.randint(0, len(objects) - 1)]
                    random_object_class = None

                    # Check if the predicate is a data property
                    is_data_property = (random_predicate, RDF.type, OWL.DatatypeProperty) in graph
                    if is_data_property:
                        data_prop_flag = True
                        random_object_class = random_object.datatype
                    else:
                        data_prop_flag = False

                    if not data_prop_flag:
                        classes_of_object_node = self.get_classes_of_current_node(graph, random_object)
                        class_of_object_node = random.choice(classes_of_object_node)
                        if class_of_object_node in URIRef('http://www.w3.org/2002/07/owl#Class'):
                            random_object_class = random_object
                        else:
                            random_object_class = class_of_object_node

                    path_sequence.append(random_predicate)
                    path_sequence.append(random_object_class)

                    # Update the current node for the next step
                    current_node = random_object

                if len(path_sequence) > 1:
                    path_sequences.append(path_sequence)
        return path_sequences