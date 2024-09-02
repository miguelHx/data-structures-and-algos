
import unittest

from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_leaf = False
        self._prefix = '__ROOT__'

    @property
    def prefix(self):
        return ''.join(self._prefix)

    @prefix.setter
    def prefix(self, value):
        self._prefix = value

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._word_count = 0
    
    @property
    def word_count(self):
        return self._word_count

    def insert(self, word: str) -> None:
        """inserts word into the trie
        time: O(s) where s is len(word)
        space: O(1)

        Args:
            word (str): _description_
        """
        current = self.root
        letters = []
        for letter in word:
            current = current.children[letter]
            letters.append(letter)
            current.prefix = ''.join(letters)
        current.is_leaf = True
        current.prefix = ''.join(letters)
        self._word_count += 1

    def search(self, word: str) -> bool:
        """searches for word in the trie
        time: O(s) where s is length of the word
        space: O(1)

        Args:
            word (str): word to search for

        Returns:
            bool: whether or not word exists in the trie
        """
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_leaf

class TestTrie(unittest.TestCase):
    
    def test_trie_basic(self):
        words = ['deal', 'dear', 'do', 'heat', 'hen', 'he']
        t = Trie()
        for w in words:
            t.insert(w)
        self.assertEqual(t.word_count, 6)
        
        for w in words:
            self.assertTrue(t.search(w))
        
        non_existent_words = ['dor', 'donger', 'dior', 'henry', 'dearest', 'alpha', 'sigma']
        for w in non_existent_words:
            self.assertFalse(t.search(w))

if __name__ == '__main__':
    unittest.main()