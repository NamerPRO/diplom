import numpy as np

from utils.ngram import NGram


class TrieCountStorage(object):

    def __init__(self):
        self.__root = ({}, 0)

    def count(self, ngram: NGram):
        node = self.__root
        i = 0
        node[1] += 1
        while ngram[i] in node[0]:
            node[0][ngram[i]][1] += 1
            node = node[0][ngram[i]]
            i += 1
        while i < len(ngram):
            if i < len(ngram) - 1:
                node[0][ngram[i]] = ({}, 1)
                node = node[0][ngram[i]]
            else:
                node[0][ngram[i]] = 1
            i += 1

    def find(self, ngram: NGram):
        counts = [-1 for _ in range(len(ngram))]
        node = self.__root
        for i in range(len(ngram) - 1):
            if ngram[i] not in node[0]:
                return counts
            counts[i] = node[ngram[i]][1]
            node = node[ngram[i]][0]
        if ngram[-1] in node:
            counts[-1] = node[ngram[-1]][1]
        return counts

    # def delete(self, ngram: NGram):
    #     node = self.__root
    #     stack = []
    #     for i in range(len(ngram)):
    #         stack.append(node)
    #         if ngram[i] not in node:
    #             return False
    #         node = node[ngram[i]]
    #     node -= 1
    #     if node == 0:
    #         j = len(ngram) - 1
    #         node = stack.pop()
    #         del node[ngram[j]]
    #         while not node and j > 0:
    #             node = stack.pop()
    #             j -= 1
    #             del node[ngram[j]]
    #     return True

class ProbabilityCounter():

    def get_probs(self):
        probs = ({}, 0)
        self.__calc_ngram_probs(self.__root, probs)
        return probs[0]

    def __calc_ngram_probs(self, node, probs):
        for word, child in node[0]:
            if type(child) is int:
                probs[0][word] = np.log(child / node[1])
            else:
                probs[0][word] = ({}, 0)
                self.__calc_ngram_probs(child, probs[0][word])
                probs[0][word][1] += np.log(child[1] / node[1])