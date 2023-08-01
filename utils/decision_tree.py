from collections import OrderedDict


class TreeNode:
    def __init__(self, child_left=None, child_right=None, leaf=None, value=None):
        self.child_left = child_left
        self.child_right = child_right
        self.leaf = leaf
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def __str__(self):
        return self._in_order_traversal(self.root)

    def _in_order_traversal(self, node):
        if node is None:
            return " "
        left = self._in_order_traversal(node.child_left)
        value = str(node.value) + " "
        right = self._in_order_traversal(node.child_right)
        return left + value + right

    def _splitter(self, *args) -> object:
        """
        TO DO: Improve the comment
        It finds the decision node
        :param: dictionary which contains key as classes and
                the values are mutually exclusive
        :rtype: generator which is the decision node and the class label
        """
        patterns = args[0]
        keys_copy = list(patterns.keys())
        min_prefix_length = 3

        # check if any item is there in any other list
        for key in keys_copy:
            lst = patterns[key]
            first_list = lst[:]
            common_pattern_exists = False

            new_k = patterns.keys()
            if len(new_k) > 1:
                for paths in first_list:
                    max_prefix_length = len(paths)
                    for prefix_length in range(min_prefix_length, max_prefix_length + 1, 2):
                        common_prefix = paths[:prefix_length]
                        for next_key, chk_lst in patterns.items():
                            if next_key != key:
                                list_to_compare = chk_lst[:]
                                # check if common_prefix occurs in other classes if it
                                # occurs then consider the next two elements
                                common_pattern_exists = any(sublist[:prefix_length] == common_prefix
                                                            for sublist in list_to_compare if
                                                            len(sublist) >= prefix_length)
                                if common_pattern_exists:
                                    break
                    if not common_pattern_exists:
                        longest_prefix = paths
                        class_label = key
                        yield longest_prefix, class_label
                del patterns[key]

            elif len(new_k) == 1:
                for k in new_k:
                    yield None, k

    def fit(self, *args):
        """
        Construct a binary decision tree using the random walks
        This method constructs a binary decision tree by taking input patterns and their associated classes.
        The decision tree is built using a recursive algorithm that splits the patterns at each node based on
        the concepts in the patterns.

        :param args: Variable-length argument list containing the patterns and their associated classes.
                     The first argument should be a dictionary of patterns, where each key is an identifier
                     (class to which it belongs) and the corresponding value is a list representing the
                     pattern (sequence of concepts).

        :return: The root node of the decision tree.

        Attributes
        ----------
        n_features (int): Total number of features (concepts) in all the input patterns.
        root (Node): The root node of the constructed decision tree.

        Example:
            tree = DecisionTree()

            # Input patterns and their associated classes
            patterns = {
            'ClassA': [['concept1', 'concept2', 'concept3'], ['concept4', 'concept5', 'concept6']],
            'ClassB': [['concept7', 'concept8', 'concept9'], ['concept10', 'concept11', 'concept12']]
            }
            class_labels here are ['ClassA', 'ClassB']

            # Fit the decision tree with the input patterns and classes
            tree.fit(*patterns)

            # Access the root node of the decision tree
            root_node = tree.root
        """

        if len(args) % 2 != 1:
            raise ValueError("Should contain the pattern as a dictionary")
        pattern = args[0]
        self.n_features = sum(len(value_list) for value_list in pattern.values())
        ordered_dict = OrderedDict(pattern)
        self.root = self._build_tree(ordered_dict)
        return self.root

    def _build_tree(self, *args):
        pattern = args[0]
        # find the split
        gen = self._splitter(pattern)
        # grow the tree
        resultant_tree = self._grow_tree_from_gen(gen)
        return resultant_tree

    def _grow_tree_from_gen(self, gen):
        try:
            check_if_node = []
            prefix, label = next(gen)
            if prefix not in check_if_node:
                check_if_node.append(prefix)
                if prefix is not None:
                    left = TreeNode(leaf=True, value=label)
                    right = self._grow_tree_from_gen(gen)
                    return TreeNode(child_left=left, child_right=right, value=prefix)
                else:
                    return TreeNode(leaf=True, value=label)
        except StopIteration:
            return None

    def predict(self, *args):
        """
        Compare the test set patterns with the generated Decision Tree.

        This method takes one or more patterns as input and traverses the Decision Tree
        to predict the corresponding nodeIDs and walks for each pattern in the test set.

        :param args: Variable-length argument list containing patterns to be predicted.
                     Each pattern should be provided as a dictionary, where the keys are nodeIDs,
                     and the values are lists representing walks (sequence of concepts).

        :return: A dictionary containing the prediction results for each pattern in the test set.
                 Each prediction entry is a nested dictionary containing the following information:
                 - 'pred_class': The predicted class associated with the pattern.
                 - 'Node': The nodeID in the Decision Tree where the prediction was made.
                 - 'concept': The sequence of concepts (walk) that resulted in the prediction.

        Raises:
            ValueError: If the Decision Tree is not trained.

        Example:
            tree = DecisionTree()
            # Assuming the Decision Tree is trained using the 'fit' method

            test_patterns = {
                'TestNode1': [['concept1', 'concept2', 'concept3']],
                'TestNode2': [['concept7', 'concept8', 'concept9']]
            }

            result = tree.predict(test_patterns)
            print(result)
            # Output: {{
            #              1: {'pred_class': 'ClassA', 'Node': 'TestNode1', 'concept': 'concept1-concept2-concept3'},
            #              2: {'pred_class': 'ClassB', 'Node': 'TestNode2', 'concept': 'concept7-concept8-concept9'}
            #          }

        """
        # TO DO: Change the count to keep count of the Random-walk of the same node.
        # As a single test nodes can have multiple random walks

        if not self.root:
            raise ValueError("The Decision Tree is not trained. Call 'fit' before predicting.")
        pattern = args[0]
        predicted_dict = {}
        count = 1
        for k, v in pattern.items():
            value = v[:]
            for walks in value:
                node = self.root
                while node:
                    try:
                        if node.leaf is None:
                            node_len = len(node.value)
                            walk_len = len(walks)
                            if walk_len >= node_len:
                                to_comp = walks[:node_len]
                                temp = '-'.join(to_comp)
                                node_temp = '-'.join(node.value)
                                if temp == node_temp:
                                    predicted_dict[count] = {
                                        'pred_class': self._get_node_value(node.child_left),
                                        'concept': temp,
                                        'Node': k
                                    }
                                    count = count + 1
                                    break
                                else:
                                    node = self._traversal(node, child='right')
                            else:
                                node = self._traversal(node, child='right')
                        else:
                            temp = '-'.join(walks)
                            predicted_dict[count] = {
                                'pred_class': self._get_node_value(node),
                                'Node': k,
                                'concept': temp
                            }
                            count = count + 1
                            break
                    except Exception as e:
                        print(f'Error{e}')
                        break
        return predicted_dict

    def _get_node_value(self, node):
        if node:
            return node.value

    def _traversal(self, node, child=None):
        if child == 'left':
            return node.child_left if node else None
        if child == 'right':
            return node.child_right if node else None
        else:
            raise ValueError('Invalid')
