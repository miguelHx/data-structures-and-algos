"""
Suppose you have a dictionary of words (you can find lists of the most common 5000 english words online, for example).
Write a function that takes in a string and returns all possible anagrams of that word (including multi-word anagrams).
For example, for "levity" it should produce "let ivy", "tel ivy", "levy it" (assuming all 3 of them are in your dictionary).


notes:  
brute force would involve calculating all the permutations of input string, of all lengths,
time complexity for this is (n)! + (n-1)! + (n-2)! + ... + (n-k)!
which is O(n! * n)

For each permutation, check if exists in dictionary, this operation is O(1)
if so, add to a list called words_found

next, try all combinations of found words, check if each is an anagram of original input
if so, add to output list
time complexity is (m choose r) * m where m is size of found words
since we check if word is anagram, which takes O(s*log(s)),
time complexity becomes (m choose r) * m * s * log(s))

overall, this becomes O(n! * n + (m choose r) * m * s * log(s)) => O(n! * n) for brute force
space is O(m) at most

todo next:  come up with a way to improve the runtime
"""


import unittest
from typing import List, Set
from itertools import permutations
from itertools import combinations


def is_anagram(s1, s2):
    if len(s1) != len(s2):
        return False
    return sorted(s1) == sorted(s2)

def find_anagrams_brute_force(s: str, dictionary: Set[str]) -> Set[str]:
    found_dict_words = []
    output = set()
    for i in range(1, len(s)):
        for p in permutations(s, i):
            ps = ''.join(p)
            if ps in dictionary:
                found_dict_words.append(ps)
    for j in range(1, len(found_dict_words)+1):
        for cmb in combinations(found_dict_words, j):
            combined = ''.join(cmb)
            if is_anagram(s, combined):
                output.add(combined)
    return output

class TestZeroMatrixFunction(unittest.TestCase):

    def setUp(self):
        self.cases = [
            (
                'levity',
                set(['let', 'ivy', 'tel', 'it', 'levy']),
                set(['letivy', 'itlevy', 'ivytel'])
            )
        ]

    def test_find_anagrams_brute_force(self):
        for s, d, expected in self.cases:
            self.assertEqual(find_anagrams_brute_force(s, d), expected, msg=(s, d, expected))

if __name__ == '__main__':
    unittest.main()