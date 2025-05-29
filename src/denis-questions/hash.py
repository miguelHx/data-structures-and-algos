"""
Hashmap implementation exercise:
Use an array to represent a memory range.
Start with a small array (like 32 slots), then double it
when the hash becomes full.
Support inserting and removing elements from the hash.
Choose a collision resolution mechanism
(either linked list or next-empty-slot)
and implement that.
Use a simple mathematical hashing function. Read up desirable
properties of hashing functions.
Start with only being able to hash integer keys, but then extend
to chars and strings.
Read up on python hashable objects, try implementing a hashable
list of integers type.
"""
import unittest

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashTable:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _rehash(self):
        """
        double capacity when hashtable is full
        """
        new_ht = HashTable(self.capacity * 2)
        for node in self.table:
            if node:
                new_ht.insert(node.key, node.value)
        self.capacity *= 2
        self.table = new_ht.table

    def insert(self, key, value):
        index = self._hash(key)

        if self.table[index] is None:
            self.table[index] = Node(key, value)
            self.size += 1
        else:
            # collision
            current = self.table[index]
            while current:
                if current.key == key:
                    current.value = value
                    return
                current = current.next
            new_node = Node(key, value)
            new_node.next = self.table[index]
            self.table[index] = new_node
            self.size += 1
        if self.size == self.capacity:
            self._rehash()

    def search(self, key):
        index = self._hash(key)

        current = self.table[index]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        raise KeyError(key)
    
    def remove(self, key):
        index = self._hash(key)

        previous = None
        current = self.table[index]

        while current:
            if current.key == key:
                if previous:
                    previous.next = current.next
                else:
                    self.table[index] = current.next
                self.size -= 1
                return
            previous = current
            current = current.next
        raise KeyError(key)
    
    def __str__(self):
        elements = []
        for i in range(self.capacity):
            current = self.table[i]
            while current:
                elements.append((current.key, current.value))
                current = current.next
        return str(elements)
    
    def __len__(self):
        return self.size
    
    def __contains__(self, key):
        try:
            self.search(key)
            return True
        except KeyError:
            return False


class TestHash(unittest.TestCase):
    
    def test_hashtable(self):
        ht = HashTable(5)

        ht.insert("apple", 3)
        ht.insert("banana", 2)
        ht.insert("cherry", 5)

        # Check if the hash table
        # contains a key
        self.assertTrue("apple" in ht)
        self.assertFalse("durian" in ht)

        # Get the value for a key
        self.assertEqual(ht.search("banana"), 2)

        # Update the value for a key
        ht.insert("banana", 4)
        self.assertEqual(ht.search("banana"), 4)

        self.assertEqual(len(ht), 3)
        ht.remove("apple")
        # Check the size of the hash table
        self.assertEqual(len(ht), 2)
    
    def test_hash(self):
        ht = HashTable(5)
        self.assertEqual(ht._hash(1), 1)
        self.assertEqual(ht._hash(6), 1)
        self.assertEqual(ht._hash(2), 2)
        self.assertEqual(ht._hash(7), 2)
        self.assertEqual(ht._hash(0), 0)
        self.assertEqual(ht._hash(5), 0)

    def test_insert_with_collision(self):
        ht = HashTable(4)
        self.assertEqual(len(ht), 0)
        ht.insert(0, 'test')
        self.assertEqual(len(ht), 1)
        self.assertEqual(str(ht), "[(0, 'test')]")
        ht.insert(1, 'test')
        self.assertEqual(len(ht), 2)
        self.assertEqual(str(ht), "[(0, 'test'), (1, 'test')]")
        ht.insert(4, 'test2')
        self.assertEqual(len(ht), 3)
        self.assertEqual(str(ht), "[(4, 'test2'), (0, 'test'), (1, 'test')]")

    def test_search(self):
        ht = HashTable(4)
        ht.insert(0, 'test')
        self.assertEqual(ht.search(0), 'test')
        ht.insert(1, 'test2')
        self.assertEqual(ht.search(1), 'test2')
        self.assertRaises(KeyError, ht.search, 3)
    
    def test_remove(self):
        ht = HashTable(4)
        ht.insert(0, 0)
        ht.insert(1, 1)
        self.assertEqual(len(ht), 2)
        ht.remove(1)
        self.assertEqual(len(ht), 1)
        self.assertRaises(KeyError, ht.remove, 3)
    
    def test_rehash(self):
        ht = HashTable(4)
        ht.insert(0, 0)
        ht.insert(1, 1)
        ht.insert(2, 2)
        self.assertEqual(ht.capacity, 4)
        self.assertEqual(len(ht), 3)
        self.assertEqual(len(ht.table), 4)
        ht.insert(3, 3)
        self.assertEqual(ht.capacity, 8)
        self.assertEqual(len(ht), 4)
        self.assertEqual(len(ht.table), 8)


if __name__ == '__main__':
    unittest.main()