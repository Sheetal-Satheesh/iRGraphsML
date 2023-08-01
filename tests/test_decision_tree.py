import unittest
from utils.decision_tree import DecisionTreeClassifier, TreeNode


class TestDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        # Initialize the DecisionTreeClassifier with default parameters
        self.tree = DecisionTreeClassifier()

    def test_fit(self):
        # Test fit method with valid input
        patterns = {
            'ClassA': [['concept1', 'concept2', 'concept3'], ['concept4', 'concept5', 'concept6']],
            'ClassB': [['concept7', 'concept8', 'concept9'], ['concept10', 'concept11', 'concept12']]
        }

        self.tree.fit(patterns)
        self.assertIsNotNone(self.tree.root)

        # Test fit method with invalid input (should raise some Error)
        # TO DO


    def test_splitter(self):
        # Test splitter method with valid input
        patterns = {
            'ClassA': [['concept1', 'concept2', 'concept3'], ['concept4', 'concept5', 'concept6']],
            'ClassB': [['concept7', 'concept8', 'concept9'], ['concept10', 'concept11', 'concept12']]
        }
        gen = self.tree._splitter(patterns)
        prefix, label = next(gen)
        self.assertEqual(prefix, ['concept1', 'concept2', 'concept3'])
        self.assertEqual(label, 'ClassA')

    def test_grow_tree_from_gen(self):
        # Test grow_tree_from_gen method with valid input
        patterns = {
            'ClassA': [['concept1', 'concept2', 'concept3'], ['concept4', 'concept5', 'concept6']],
            'ClassB': [['concept7', 'concept8', 'concept9'], ['concept10', 'concept11', 'concept12']]
        }
        gen = self.tree._splitter(patterns)
        tree = self.tree._grow_tree_from_gen(gen)
        self.assertIsNotNone(tree)
        self.assertIsInstance(tree, TreeNode)

    def test_predict(self):
        # Test predict method with a trained decision tree and test patterns
        patterns = {
            'ClassA': [['concept1', 'concept2', 'concept3'], ['concept4', 'concept5', 'concept6']],
            'ClassB': [['concept7', 'concept8', 'concept9'], ['concept10', 'concept11', 'concept12']]
        }

        self.tree.fit(patterns)

        test_patterns = {
            'TestNode1': [['concept1', 'concept2', 'concept3']],
            'TestNode2': [['concept7', 'concept8', 'concept9']]
        }

        result = self.tree.predict(test_patterns)
        self.assertEqual(len(result), 2)

        # Ensure that the result dictionary has the correct format
        self.assertDictEqual(
            result,
            {
                1: {'pred_class': 'ClassA', 'Node': 'TestNode1', 'concept': 'concept1-concept2-concept3'},
                2: {'pred_class': 'ClassB', 'Node': 'TestNode2', 'concept': 'concept7-concept8-concept9'}
            }
        )

if __name__ == '__main__':
    unittest.main()