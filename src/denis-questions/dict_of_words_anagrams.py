"""
Suppose you have a dictionary of words (you can find lists of the most common 5000 english words online, for example).
Write a function that takes in a string and returns all possible anagrams of that word (including multi-word anagrams).
For example, for "levity" it should produce "let ivy", "tel ivy", "levy it" (assuming all 3 of them are in your dictionary).
"""

import copy
import sys
import unittest
from typing import List, Set, Dict
from itertools import permutations, combinations, product
from collections import Counter, defaultdict
from .trie import TrieNode, Trie


def _is_anagram(s1: str, s2: str) -> bool:
    """checks if s1 is anagram of s2
    time: O(n)
    space: O(n)

    Args:
        s1 (str): string to compare
        s2 (str): 2nd string to compare

    Returns:
        bool: whether or not s1 is anagram of s2
    """
    if len(s1) != len(s2):
        return False
    return Counter(s1.lower()) == Counter(s2.lower())

def _find_anagram_combos(s: str, lst: List[str]) -> Set[str]:
    """runs all possible anagrams from combos of lst and s
    time:
        O((n choose k) * s)
    space:
        O(n choose k)

    Args:
        s (str): string to find anagrams for
        lst (List[str]): list of words with potential anagram combos

    Returns:
        Set[str]: found anagram combos
    """
    output = set()
    for j in range(1, len(lst)+1):
        for cmb in combinations(lst, j):
            cs = ''.join(cmb)
            if _is_anagram(s, cs):
                output.add(cs)
    return output


# arr - array to store the combination
# index - next location in array
# num - given number
# reducedNum - reduced number 
def findCombinationsUtil(arr, index, num, reducedNum):
    output = []
    # Base condition
    if (reducedNum < 0):
        return []

    # If combination is 
    # found, print it
    if (reducedNum == 0):
        combo_numbers = []
        for i in range(index):
            combo_numbers.append(arr[i])
        return [combo_numbers]

    # Find the previous number stored in arr[]. 
    # It helps in maintaining increasing order
    prev = 1 if(index == 0) else arr[index - 1]

    # note loop starts from previous 
    # number i.e. at array location
    # index - 1
    for k in range(prev, num + 1):
        # next element of array is k
        arr[index] = k

        # call recursively with
        # reduced number
        output.extend(findCombinationsUtil(arr, index + 1, num, reducedNum - k))
    return output

# Function to find out all 
# combinations of positive numbers 
# that add upto given number.
# It uses findCombinationsUtil() 
def findCombinations(n):
    # array to store the combinations
    # It can contain max n elements
    arr = [0] * n
    # find all combinations
    return findCombinationsUtil(arr, 0, n, n)

def _find_anagram_combos_optimized(s: str, words: List[str], combo_sets: List[List[int]]) -> Set[str]:
    """finds all possible combos that add up to length of s.
    this algo will put words into buckets associated with
    their lengths. Then will iterate over the combo_sets list,
    using each number in subarrays to key the bucket and check
    for anagrams
    
    Runtime:
        O(2^s * Product of b[i] in k arrays * s * l * log(l))
        where s is the length of input string s,
        k is the average length of combo subsets,
        b[i] is the number of words in each bucket
        grouped by word length,
        and l is the average length of each cartesian product
        check operation
    Space: O(w) where w is # of words in dictionary

    Args:
        s (str): string to find subset word anagrams of
        lst (List[str]): list of words in s
        combo_sets (List[List[int]]): list of combos that add up to len(s)

    Returns:
        Set[str]:
            list of anagrams found in s from words in lst.
            including multi-word anagrams
    """
    anagrams = set()
    buckets = defaultdict(list)
    for w in words:
        buckets[len(w)].append(w)
    
    for curr_combo_set in combo_sets:
        s_copy = copy.copy(s)
        curr_combo_buckets = [buckets[num] for num in curr_combo_set]
        # cartesian product look for anagrams
        for cartesian_product_as_tuple in product(*curr_combo_buckets):
            combo = ''.join(map(str, sorted(cartesian_product_as_tuple)))
            if _is_anagram(s, combo):
                anagrams.add(combo)
    return anagrams
        

def dfs_trie(root: TrieNode, pool: Dict[str, int]) -> List[str]:
    """similar to a graph dfs, we will
    be traversing through the trie, looking for
    words that contain letters in our letter pool
    time:
        O(p), where p is pool size (sum of all values in pool dict)
    space:
        O(w), where w is # of words in trie tree

    Args:
        root (TrieNode): starting node to traverse from
        pool (Dict[str, int]):
            letters with counts which represent possible
            dfs layers to traverse through

    Returns:
        List[str]: list of found words given the pool
    """
    if root is None:
        raise TypeError
    found_words = []
    if root.is_leaf:
        found_words.append(root.prefix)
    for letter, trie_node in root.children.items():
        if pool[letter] > 0:
            pool[letter] -= 1
            found_words.extend(dfs_trie(trie_node, pool))
            pool[letter] += 1
    return found_words


def find_words_with_trie(s: str, root: TrieNode) -> List[str]:
    """this function will find all possible words
    from s that may exist in the dictionary as a trie.
    we will basically do a tree search for all possible
    letter sequences for each letter in s
    time: O(s)
    space: O(s + w)

    Args:
        s (str): the string with possible words in dictionary
        root (TrieNode): root trie node for searching

    Returns:
        List[str]: list of words found in s
    """
    pool = Counter(s)
    return dfs_trie(root, pool)


def find_anagrams_with_trie(s: str, root: TrieNode) -> Set[str]:
    """semi brute force method, finds all the potential dict words
    from s, then brute forces potential anagram combos.
    time:
        O(s + (n choose k) * s)
    space complexity would be O(w + (n choose k))

    Args:
        s (str): string with potential dict word anagrams (including multi-word)
        root (TrieNode): root of trie dictionary tree

    Returns:
        Set[str]:
            words found from dict that are anagrams of input s,
            including multiword anagrams
    """
    found_dict_words = find_words_with_trie(s, root)
    return _find_anagram_combos(s, found_dict_words)

def find_anagrams_brute_force(s: str, dictionary: Set[str]) -> Set[str]:
    """brute force would involve calculating all the permutations of input string,
    of all lengths.
    time:
        two parts, first part (all permuations and check if in dict):
            (n)! + (n-1)! + (n-2)! + ... + (n-k)!, where k is 1 up to n
            => O(n! * n)
        second part, all combos of dict words, check if anagram of s:
            O(m choose k) * m where m is size of found words, k is 1 up to m
            for each combo, check if word is anagram, which takes O(s),
            => O(m choose k) * m * s)
        overall time:
            O(n! * n + (m choose k) * m * s)

    space:
        O(w), where w is the number of words in dict

    Args:
        s (str): string with potential dict word anagrams
        dictionary (Set[str]): set of dictionary words

    Returns:
        Set[str]:
            words found from dict that are anagrams of input s,
            including multiword anagrams
    """
    found_dict_words = []
    output = set()
    for i in range(1, len(s)+1):
        for p in permutations(s, i):
            ps = ''.join(p)
            if ps in dictionary:
                found_dict_words.append(ps)
    return _find_anagram_combos(s, found_dict_words)

class TestFindAnagrams(unittest.TestCase):

    def setUp(self):
        self.is_anagram_cases = [
            ('listen', 'silent', True),
            ('bag', 'dag', False),
            ('', '', True),
            ('racecar', 'racecar', True),
            ('racecar', 'carrace', True),
            ('test', 'tttt', False),
            ('i', 't', False)
        ]
        self.find_anagrams_combos_cases = [
            (
                'levity',
                ['it', 'let', 'ivy', 'tel', 'levi', 'levy'],
                set(['letivy', 'itlevy', 'ivytel'])
            )
        ]
        self.find_anagrams_brute_force_cases = [
            (
                'levity',
                set(['let', 'ivy', 'tel', 'it', 'levy', 'lee', 'levi']),
                set(['letivy', 'itlevy', 'ivytel'])
            ),
            (
                'levity',
                set(['let', 'ivy', 'tel', 'it', 'levy', 'lee', 'levi', 'vityle']),
                set(['letivy', 'itlevy', 'ivytel', 'vityle'])
            ),
            (
                'ytivel',
                set(['le', 'vi', 'ty', 'vel', 'tiy']),
                set(['tyvile', 'tiyvel'])
            ),
            (
                'todhae',
                set(['deal', 'dear', 'do', 'heat', 'hen', 'he']),
                set(['doheat'])
            ),
            (
                'dtoeah',
                set(['deal', 'dear', 'do', 'heat', 'hen', 'he']),
                set(['doheat'])
            )
        ]

    def test_find_anagrams_brute_force(self):
        for s, d, expected in self.find_anagrams_brute_force_cases:
            self.assertEqual(find_anagrams_brute_force(s, d), expected, msg=(s, d, expected))
    
    def test_is_anagram(self):
        for s1, s2, expected in self.is_anagram_cases:
            self.assertEqual(_is_anagram(s1, s2), expected, msg=(s1, s2, expected))
    
    def test_find_anagram_combos(self):
        for s, lst, expected in self.find_anagrams_combos_cases:
            self.assertEqual(_find_anagram_combos(s, lst), expected, msg=(s, lst, expected))

    def test_find_anagram_combos_optimized(self):
        s = 'levity'
        words = ['it', 'let', 'ivy', 'tel', 'levi', 'levy']
        expected = set(['ivylet', 'itlevy', 'ivytel'])
        combo_sets = findCombinations(len(s))
        self.assertEqual(_find_anagram_combos_optimized(s, words, combo_sets), expected, msg=(s, words))
        
        s = 'todhae'
        words = ['deal', 'dear', 'do', 'heat', 'hen', 'he', 'eat', 'tea']
        expected = set(['doheat'])
        combo_sets = findCombinations(len(s))
        self.assertEqual(_find_anagram_combos_optimized(s, words, combo_sets), expected, msg=(s, words))

    def test_find_words_with_trie(self):
        """building trie data structure ahead of time:
        O(n * m), where n is the number of words to insert,
        and m is the average length of words
        """

        words = ['deal', 'dear', 'do', 'heat', 'hen', 'he']
        t = Trie()
        for w in words:
            t.insert(w)
        self.assertEqual(
            find_words_with_trie('todhae', t.root),
            ['do', 'he', 'heat']
        )

        words = ['let', 'ivy', 'tel', 'it', 'levy', 'lee', 'levi']
        t = Trie()
        for w in words:
            t.insert(w)
        self.assertEqual(
            find_words_with_trie('levity', t.root),
            ['let', 'levy', 'levi', 'ivy', 'it', 'tel']
        )

    def test_find_anagrams_with_trie(self):
        words = ['let', 'ivy', 'tel', 'it', 'levy', 'lee', 'levi']
        t = Trie()
        for w in words:
            t.insert(w)
        anagrams = find_anagrams_with_trie('levity', t.root)
        self.assertEqual(anagrams, set(['letivy', 'levyit', 'ivytel']))

        words = ['let', 'ivy', 'tel', 'it', 'levy', 'lee', 'levi', 'vityle']
        t = Trie()
        for w in words:
            t.insert(w)
        anagrams = find_anagrams_with_trie('levity', t.root)
        self.assertEqual(anagrams, set(['letivy', 'levyit', 'ivytel', 'vityle']))

        words = ['le', 'vi', 'ty', 'vel', 'tiy']
        t = Trie()
        for w in words:
            t.insert(w)
        anagrams = find_anagrams_with_trie('ytivel', t.root)
        self.assertEqual(anagrams, set(['veltiy', 'levity']))

        words = ['deal', 'dear', 'do', 'heat', 'hen', 'he']
        t = Trie()
        for w in words:
            t.insert(w)
        anagrams = find_anagrams_with_trie('todhae', t.root)
        self.assertEqual(anagrams, set(['doheat']))

        words = ['deal', 'dear', 'do', 'heat', 'hen', 'he']
        t = Trie()
        for w in words:
            t.insert(w)
        anagrams = find_anagrams_with_trie('dtoeah', t.root)
        self.assertEqual(anagrams, set(['doheat']))

if __name__ == '__main__':
    unittest.main()