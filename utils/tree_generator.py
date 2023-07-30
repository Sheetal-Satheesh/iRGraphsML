from utils.elements import Elements

class TreeNode:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None

# TODO:
class TreeGenerator:
    def __init__(self, elements):
        self.root = None
        self.current_node = None
        self.left_child = None
        self.right_child = None
        self.construct_tree(elements)

    def construct_tree(self, elements):
        if not elements:
            return
        self.root = self.create_node(self.conjunction())
        self.current_node = self.root
        self.left_child = elements.pop(0)
        self.right_child = elements.pop(0)
        self.add_node(self.left_child, self.right_child)
        while len(elements) > 0:
            self.left_child = elements.pop(0)
            self.add_node(self.left_child)

    def insert_right(self, value=None):
        if value is not None:
            if self.current_node.right is None:
                self.current_node.right = TreeNode(value)
            else:
                old_right = self.current_node.right
                new_right = self.create_node(self.conjunction())
                new_right.left = old_right
                new_right.right = TreeNode(value)
                self.current_node.right = new_right

    def insert_left(self, value):
        if self.current_node.left is None:
            self.current_node.left = value
        else:
            return False
        return True

    def set_root_operator(self, operator):
        self.root.value = operator

    def set_root_right_child(self, value):
        self.root.right = TreeNode(value)

    def add_node(self, left_child, right_child=None):
        val = self.insert_left(left_child)
        if not val:
            self.insert_right(left_child)
        else:
            self.insert_right(right_child)

    def create_node(self, value=None):
        node = TreeNode(value)
        return node

    def conjunction(self):
        return '⊓'

    def __str__(self):
        return self.print_concept(self.root)

    def print_concept(self, node):
        result = ""
        if node:
            if node.left is not None:
                if isinstance(node.left.value, object) and isinstance(node.left.value, Elements):
                    result += '(' + str(node.left.value) + ')' + str(node.value) + '('+ self.print_concept(node.right) +')'
                else:
                    if node.left is not None:
                        result += str(node.left) + str(node.value) + '(' + self.print_concept(
                            node.right) + ')'
            else:
                result += '(' + str(node.value) + ')'
        return result
