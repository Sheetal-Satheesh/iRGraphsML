from abc import ABCMeta, abstractmethod
import random
from collections import Counter
from rdflib import URIRef, RDF, OWL, Literal

class RandomWalk(metaclass=ABCMeta):
    def __init__(self, graph, data_dict, num_walks=4, depth=3):
        self.random_walk = {}
        self.data_dict = data_dict
        self.graph = graph
        self.num_walks = num_walks
        self.depth = depth
        self.most_occurring_pattern = {}

    def get_random_walk(self):
        return self.random_walk

    def set_random_walk(self):
        self._random_walk(self.data_dict, self.graph, self.num_walks, self.depth
                          )
    def clean(self):
        self.random_walk = None

    def _random_walk(self, data_dict, graph, num_walks=5, depth=3):
        classes_for_rw = data_dict.keys()
        for cls in classes_for_rw:
            training_nodes = data_dict[cls]
            for node in training_nodes:
                path_sequences = self.generate_random_walks([node], graph, num_walks, depth)
                if node not in self.random_walk:
                    self.random_walk[node] = [path_sequences, cls]

    def calculate_object_probabilities(self, current_node, predicates, graph):
        # Calculate probabilities based on object cardinality
        object_cardinalities = Counter()

        for predicate in predicates:
            objects = list(graph.objects(subject=current_node, predicate=predicate))
            object_cardinalities[predicate] = len(objects)

        total_cardinality = sum(object_cardinalities.values())
        if total_cardinality == 0:
            # If there are no objects, assign equal probabilities to predicates
            return [1.0 / len(predicates)] * len(predicates)
        else:
            # Calculate probabilities based on cardinality
            object_probabilities = [object_cardinalities[predicate] / total_cardinality for predicate in predicates]
            return object_probabilities

    def get_classes_of_current_node(self, graph, current_node):
        # Get the classes of the current node
        class_triples = graph.triples((current_node, RDF.type, None))
        classes_of_current_node = []
        for _, _, class_ in class_triples:
            classes_of_current_node.append(class_)
        return classes_of_current_node


class BiasedRandomWalk(RandomWalk):
    def generate_random_walks(self, nodes, graph, num_walks, depth):
        path_sequences = []
        for node in nodes:
            for walk in range(num_walks):
                current_node = URIRef(node)
                classes_of_current_node = self.get_classes_of_current_node(graph, current_node)
                class_of_current_node = random.choice(classes_of_current_node)
                path_sequence = [class_of_current_node]
                for w in range(depth - 1):
                    # Get all outgoing predicates from the current node
                    predicates = list(graph.predicates(subject=current_node))
                    predicates = [predicate for predicate in predicates if predicate != RDF.type]
                    if not predicates:
                        break  # Stop the random walk if there are no outgoing predicates

                    # Calculate probabilities for objects based on cardinality
                    object_probabilities = self.calculate_object_probabilities(current_node, predicates, graph)

                    # Choose the predicate with the highest probability
                    random_predicate = predicates[object_probabilities.index(max(object_probabilities))]

                    # Get all objects connected by the random predicate
                    objects = list(graph.objects(subject=current_node, predicate=random_predicate))

                    if not objects:
                        break

                    # Choose a random object
                    random_object = objects[random.randint(0, len(objects) - 1)]
                    random_object_class = None

                    # Check if the object is a literal
                    if isinstance(random_object, Literal):
                        data_prop_flag = True
                        random_object_class = random_object
                    else:
                        data_prop_flag = False

                    if not data_prop_flag:
                        # print('Current', current_node)
                        # print('Random', random_object)
                        classes_of_object_node = self.get_classes_of_current_node(graph, random_object)
                        if len(classes_of_object_node) > 0:
                            # print('Class of Object Node', classes_of_object_node)
                            class_of_object_node = random.choice(classes_of_object_node)
                            if class_of_object_node in URIRef('http://www.w3.org/2002/07/owl#Class'):
                                random_object_class = random_object
                            else:
                                random_object_class = class_of_object_node
                        else:
                            random_object_class = None

                    path_sequence.append(random_predicate)
                    path_sequence.append(random_object_class)

                    # Update the current node for the next step
                    current_node = random_object

                # if len(path_sequence) >= ((depth * 2) - 1):
                #     path_sequences.append(path_sequence)
                if 1 < len(path_sequence) <= ((depth * 2) - 1):
                    path_sequences.append(path_sequence)
        return path_sequences


class RandomWalkWithoutBias(RandomWalk):
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
                    if not predicates:
                        break  # Stop the random walk if there are no outgoing predicates

                    # Choose a random edge/predicate
                    random_predicate = predicates[random.randint(0, len(predicates) - 1)]
                    # print('Choosen Predicate', random_predicate)

                    # Get all objects connected by the random predicate
                    objects = list(graph.objects(subject=current_node, predicate=random_predicate))
                    # print('Object', objects)

                    if not objects:
                        break

                    # Choose a random object
                    random_object = objects[random.randint(0, len(objects) - 1)]
                    # print('Choosen Object', random_object)
                    random_object_class = None

                    # Check if the object is a literal
                    if isinstance(random_object, Literal):
                        data_prop_flag = True
                        random_object_class = random_object
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

                if 1 < len(path_sequence) <= ((depth * 2) - 1):
                    path_sequences.append(path_sequence)
        return path_sequences
