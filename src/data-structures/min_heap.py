import unittest


class MinHeap:
    def __init__(self):
        self.a = []
    
    def insert(self, val):
        self.a.append(val)
        i = len(self.a) - 1
        while i > 0 and self.a[(i-1) // 2] > self.a[i]:
            # swap
            self.a[i], self.a[(i-1) // 2] = self.a[(i-1) // 2], self.a[i]
            i = (i - 1) // 2 # parent
    
    def delete(self, value):
        i = -1
        for j in range(len(self.a)):
            if self.a[j] == value:
                i = j
                break
        if i == -1:
            return
        self.a[i] = self.a[-1]
        self.a.pop()
        self.minHeapify(i, len(self.a))
    
    def minHeapify(self, i , n):
        smallest = i
        left = 2*i + 1
        right = 2*i + 2

        if left < n and self.a[left] < self.a[smallest]:
            smallest = left
        if right < n and self.a[right] < self.a[smallest]:
            smallest = right
        if smallest != i:
            self.a[i], self.a[smallest] = self.a[smallest], self.a[i]
            self.minHeapify(smallest, n)

    def search(self, element):
        for j in self.a:
            if j == element:
                return True
        return False

    def getMin(self):
        return self.a[0] if self.a else None
    
    def printHeap(self):
        print("Min Heap: ", self.a)

class TestMinHeap(unittest.TestCase):
    def test_min_heap(self):
        h = MinHeap()
        values = [10, 7, 11, 5, 4, 13]
        for value in values:
            h.insert(value)
        h.printHeap()
        self.assertEqual(h.a, [4, 5, 11, 10, 7, 13])

        h.delete(7)
        print('heap after deleting 7: ', h.a)
        self.assertEqual(h.a, [4, 5, 11, 10, 13])

        self.assertTrue(h.search(10))
        self.assertFalse(h.search(999))
        self.assertTrue(h.getMin(), 4)

if __name__ == '__main__':
    unittest.main()