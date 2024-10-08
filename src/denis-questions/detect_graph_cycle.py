"""
Write a function that detects if a graph has any cycles in it.
"""


import unittest

from collections import deque
from dataclasses import dataclass
from typing import List, Deque, Set

@dataclass
class Node:
    id: int
    children: 'List[Node]'
    visited: bool = False

    def add_child(self, *nodes: 'Node'):
        for node in nodes:
            self.children.append(node)

    def print_children(self):
        print('Adjacency list for node {}: {}'.format(self.id, ', '.join(str(child.id) for child in self.children)))

    def __str__(self):
        return f'Node ({self.id})'


@dataclass
class Graph:
    nodes: 'List[Node]'

    def print_graph(self):
        for node in self.nodes:
            node.print_children()

    def reset_visited(self):
        for node in self.nodes:
            node.visited = False

    def detect_cycles_util(self, n: Node, visited: Set, rec_stack: Set) -> bool:
        """detects cycles in a graph given a starting node.
        other arguments for sharing data throughout recursive calls
        time:
            O(N + C) where N is the # of nodes in the graph
            and C is the # of children per node, aka edges in adjacency list
        space:
            O(N)

        Args:
            n (Node): node from graph
            visited (Set): visited nodes
            rec_stack (Set): ancestor nodes

        Returns:
            bool: whether or not a cycle exists
        """
        if n.id not in visited:
            visited.add(n.id)
            rec_stack.add(n.id)

            for c in n.children:
                if c.id not in visited and self.detect_cycles_util(c, visited, rec_stack):
                    return True
                elif c.id in rec_stack:
                    return True
        rec_stack.remove(n.id)
        return False

    def detect_cycles_rec(self) -> bool:
        """detects cycles in a graph.
        The graph can have separated forests
        time:
            O(N + C) where N is the # of nodes in the graph
            and C is the # of children per node, aka edges in adjacency list
        space:
            O(N)

        Returns:
            bool: whether or not a cycle exists
        """
        visited = set()
        rec_stack = set()
        for n in self.nodes:
            if n.id not in visited and self.detect_cycles_util(n, visited, rec_stack):
                return True
        return False

    def detect_cycles_iter(self) -> bool:
        """iterative version of detect_cycles_rec.
        Same time and space complexity.

        Returns:
            bool: whether a cycle exists
        """
        visited = set()
        for node in self.nodes:
            if node.id in visited:
                continue
            stack = [node]
            ancestors = set()
            while stack:
                n = stack.pop()
                if n.id not in visited:
                    ancestors.add(n.id)
                    visited.add(n.id)
                    stack.append(n)
                    for c in n.children:
                        if c.id not in visited:
                            stack.append(c)
                        elif c.id in ancestors:
                            return True
                else:
                    ancestors.discard(n.id)

        return False

def dfs(root: Node) -> List[int]:
    """DFS recursive
    takes in a root, returns a list
    of ids of the sequence of visited
    nodes.

    Args:
        root (Node): starting node

    Returns:
        List[int]: list of node IDs (i.e. [0, 1, 3])
    """
    if root is None:
        raise TypeError
    visited_list: List[int] = [root.id]
    root.visited = True
    for node in root.children:
        if not node.visited:
            visited_list.extend(dfs(node))
    return visited_list

def dfs_iter(root: Node) -> List[int]:
    """DFS non recursive

    Args:
        root (Node): starting node
        
    Returns:
        List[int]: list of node IDs (i.e. [0, 1, 3])
    """
    if root is None:
        raise TypeError
    visited_list = []
    stack = [root]
    while stack:
        n = stack.pop()
        n.visited = True
        visited_list.append(n.id)
        for c in n.children:
            if not c.visited:
                stack.append(c)
    return visited_list


def bfs(root: Node) -> List[int]:
    """Simple BFS.
    takes in a root, returns a list
    of ids of the sequence of visited
    nodes.

    Args:
        root (Node): starting node

    Returns:
        List[int]: List[int]: list of node IDs (i.e. [0, 1, 4])
    """
    if root is None:
        raise TypeError
    visited_list: List[int] = [root.id]
    root.visited = True
    queue: Deque[Node] = deque([root])
    while queue:
        node = queue.popleft()
        for n in node.children:
            if not n.visited:
                n.visited = True
                visited_list.append(n.id)
                queue.append(n)
    return visited_list


class TestGraph(unittest.TestCase):

    def test_detect_cycles(self):
        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n0.add_child(n1, n2)
        n1.add_child(n2)
        n2.add_child(n0, n3)
        g = Graph([n0, n1, n2, n3])
        self.assertTrue(g.detect_cycles_iter())
        self.assertTrue(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n0.add_child(n1, n4, n5)
        n1.add_child(n3, n4)
        n3.add_child(n2, n4)
        g = Graph([n0, n1, n2, n3, n4, n5])
        self.assertFalse(g.detect_cycles_iter())
        self.assertFalse(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n0.add_child(n1)
        n1.add_child(n2)
        n2.add_child(n3)
        n3.add_child(n4)
        n4.add_child(n3)
        g = Graph([n0, n1, n2, n3, n4])
        self.assertTrue(g.detect_cycles_iter())
        self.assertTrue(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n6 = Node(6, [])
        n7 = Node(7, [])
        n0.add_child(n1, n2)
        n1.add_child(n3, n4)
        n2.add_child(n5, n6)
        n6.add_child(n7)
        n7.add_child(n0)
        g = Graph([n0, n1, n2, n3, n4, n5, n6, n7])
        self.assertTrue(g.detect_cycles_iter())
        self.assertTrue(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n6 = Node(6, [])
        n7 = Node(7, [])
        n0.add_child(n1, n2)
        n1.add_child(n3, n4)
        n2.add_child(n1)
        g = Graph([n0, n1, n2, n3, n4])
        self.assertFalse(g.detect_cycles_iter())
        self.assertFalse(g.detect_cycles_rec())

        # separated trees
        # 1. both no cycles,
        # 2. both cycles,
        # 3. one with cycle, one w/o
        n0 = Node(0, [])
        n1 = Node(1, [n0])
        n2 = Node(2, [n1])

        n3 = Node(3, [])
        n4 = Node(4, [n3])
        n5 = Node(5, [n4])
        g = Graph([n0, n1, n2, n3, n4, n5])
        self.assertFalse(g.detect_cycles_iter())
        self.assertFalse(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [n0])
        n2 = Node(2, [n1])
        n0.add_child(n2)

        n3 = Node(3, [])
        n4 = Node(4, [n3])
        n5 = Node(5, [n4])
        n3.add_child(n5)
        g = Graph([n0, n1, n2, n3, n4, n5])
        self.assertTrue(g.detect_cycles_iter())
        self.assertTrue(g.detect_cycles_rec())

        n0 = Node(0, [])
        n1 = Node(1, [n0])
        n2 = Node(2, [n1])

        n3 = Node(3, [])
        n4 = Node(4, [n3])
        n5 = Node(5, [n4])
        n3.add_child(n5)
        g = Graph([n0, n1, n2, n3, n4, n5])
        self.assertTrue(g.detect_cycles_iter())
        self.assertTrue(g.detect_cycles_rec())

    def test_basic_graph_creation(self):
        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n6 = Node(6, [])
        n0.add_child(n1)
        n1.add_child(n2)
        n2.add_child(n0, n3)
        n3.add_child(n2)
        n4.add_child(n6)
        n5.add_child(n4)
        n6.add_child(n5)
        nodes = [n0, n1, n2, n3, n4, n5, n6]
        g = Graph(nodes)
        # g.print_graph()

    def test_dfs(self):
        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n0.add_child(n1, n4, n5)
        n1.add_child(n3, n4)
        n3.add_child(n2, n4)
        result: List[int] = dfs(n0)
        self.assertEqual(result, [0, 1, 3, 2, 4, 5])

    def test_dfs_iter(self):
        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n6 = Node(6, [])
        n7 = Node(7, [])
        n0.add_child(n1, n4, n5)
        n1.add_child(n3, n4)
        n3.add_child(n2, n4)
        self.assertEqual(dfs_iter(n0), [0, 5, 4, 1, 3, 2])
        n4.add_child(n6)
        n5.add_child(n7)
        g = Graph([n0, n1, n2, n3, n4, n5, n6, n7])
        g.reset_visited()
        self.assertEqual(dfs_iter(n0), [0, 5, 7, 4, 6, 1, 3, 2])

    def test_bfs(self):
        n0 = Node(0, [])
        n1 = Node(1, [])
        n2 = Node(2, [])
        n3 = Node(3, [])
        n4 = Node(4, [])
        n5 = Node(5, [])
        n6 = Node(6, [])
        n7 = Node(7, [])
        n0.add_child(n1, n4, n5)
        n1.add_child(n3, n4)
        n3.add_child(n2, n4)
        self.assertEqual(bfs(n0), [0, 1, 4, 5, 3, 2])
        n4.add_child(n6)
        n5.add_child(n7)
        g = Graph([n0, n1, n2, n3, n4, n5, n6, n7])
        g.reset_visited()
        self.assertEqual(bfs(n0), [0, 1, 4, 5, 3, 6, 7, 2])
    
    def test_raise_type_error(self):
        self.assertRaises(TypeError, dfs, None)
        self.assertRaises(TypeError, dfs_iter, None)
        self.assertRaises(TypeError, bfs, None)


if __name__ == '__main__':
    unittest.main()