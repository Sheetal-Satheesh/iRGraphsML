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
        The binary decision tree
        Takes input the patterns with which we construct
        a decision tree
        :return: the root node of the decision tree

        Attributes
        ----------
        Dict: which contains the pattern

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
        r"""
        Compare the test set with the generated <Decision Tree>
        :return: a dictionary containing the prediction with the nodeID
                and walk
        """
        if not self.root:
            raise ValueError("The Decision Tree is not trained. Call 'fit' before predicting.")
        pattern = args[0]
        predicted_dict = {}
        count = 1
        node = self.root
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
