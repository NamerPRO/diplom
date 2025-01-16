

class TrieNode(object):

    def __init__(self, value, parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        self.count = 0


class Trie(object):

    def __init__(self):
        self.root = TrieNode("<R>")

    def insert(self, word):
        data = self.__descent(word, self.root)
        node = data[1]
        if not data[0]:
            for i in range(len(word) - data[2]):
                successor = TrieNode(word[i + data[2]], parent=node)
                node.children.append(successor)
                node = successor
        node.count += 1

    def __descent(self, word, node: TrieNode):
        for i in range(len(word)):
            successor = None
            for child in node.children:
                if word[i] == child.value:
                    successor = child
                    break
            if successor is None:
                return False, node, i
            node = successor
        return True, node

    def find(self, word):
        data = self.__descent(word, self.root)
        if not data[0]:
            return None
        return data[1].count

    def delete(self, word):
        data = self.__descent(word, self.root)
        if not data[0]:
            return False
        node = data[1]
        node.count -= 1
        while node.count == 0 and node.parent is not None and not node.children:
            parent = node.parent
            for i in range(len(parent.children)):
                if parent.children[i] == node:
                    del parent.children[i]
                    break
        return True