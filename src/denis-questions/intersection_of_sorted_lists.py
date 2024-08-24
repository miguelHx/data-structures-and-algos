"""
Suppose I have two sorted lists of numbers. Find the intersection of those lists in O(n) time. Extend this approach to work over an arbitrary number of lists.
"""

import unittest
from typing import List


def find_intersection(l1: List[int], l2: List[int]) -> List[int]:
    """same desc as above
    time: O(n)
    space: O(n)
    
    Args:
        l1 (List[int]): sorted list
        l2 (List[int]): sorted list

    Returns:
        List[int]: intesection
    """
    res = []
    i, j = 0, 0
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            i += 1
        elif l2[j] < l1[i]:
            j += 1
        else:
            res.append(l1[i])
            i += 1
            j += 1
    return res

def find_intersection_extended(lists: List[List[int]]) -> List[int]:
    """same as above, extended for arbitrary # of lists
    time: O(m * n), where m is length of lists, and n is length of longest sublist
    space: O(n), where n is longest sublist

    Args:
        lists (List[List[int]]): list of lists

    Returns:
        List[int]: intersection of all lists
    """
    inter = []
    while len(lists) > 1:
        inter = find_intersection(lists.pop(), lists.pop())
        if inter == []:
            return []
        lists.append(inter)
    return inter       


class TestFindIntersectionFunction(unittest.TestCase):

    def setUp(self):
        self.two_list_cases = [
            (
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [15, 16, 17, 38],
                []
            ),
            (
                [1, 2, 3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8]
            ),
            (
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [5, 6, 9, 10, 12, 13, 14],
                [5, 6, 9, 10]
            ),
            (
                [1, 4, 7, 19, 20],
                [2, 3, 6, 7, 12, 20],
                [7, 20]
            ),
            (
                [1, 4, 7, 19, 21],
                [21],
                [21]
            ),
            (
                [],
                [],
                []
            ),
            (
                [1],
                [0],
                []
            ),
            (
                [1],
                [1],
                [1]
            ),
            (
                [7, 7, 7, 7, 8],
                [1, 2, 7, 7, 8, 9],
                [7, 7, 8]
            ),
            (
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ),
            (
                [1, 1, 1],
                [1, 1],
                [1, 1]
            )
        ]
        self.n_list_cases = [
            (
                [
                    [1, 2, 3, 4, 5, 6, 7],
                    [3, 4, 5, 6],
                    [4, 5, 9, 12]
                ],
                [4, 5]
            ),
            (
                [
                    [2, 3, 4, 7, 12, 59],
                    [1, 5, 6, 12, 100],
                    [2, 4, 6, 12, 99],
                    [12]
                ],
                [12]
            ),
            (
                [
                    [9, 8, 7]
                ],
                []
            ),
            (
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3],
                    [99, 100, 101],
                    [6, 7, 8],
                    [55, 66, 77]
                ],
                []
            ),
            (
                [
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                ],
                [1, 2, 3, 4, 5, 6, 7]
            ),
            (
                [
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5, 6, 7],
                ],
                [1, 2, 3, 5, 6, 7]
            ),
            (
                [
                    [],
                    [],
                    [],
                    []
                ],
                []
            ),
            (
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                [0, 0, 0, 0, 0]
            ),
            (
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0],
                    [0, 0],
                    [0]
                ],
                [0]
            ),
            (
                [
                    [1],
                    [1, 1],
                    [1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ],
                [1]
            )
        ]

    def test_find_intersection(self):
        for l1, l2, expected in self.two_list_cases:
            self.assertEqual(find_intersection(l1, l2), expected, msg=(l1, l2, expected))

    def test_find_intersection_extended(self):
        for l, expected in self.n_list_cases:
            self.assertEqual(find_intersection_extended(l), expected, msg=(l, expected))

if __name__ == '__main__':
    unittest.main()