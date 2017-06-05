import re

import pandas as pd
import numpy as np
from sklearn.utils import deprecated

from load_lm import LanguageModel
import itertools


@DeprecationWarning
class OldHmmNgramRecognizer():

    def __init__(self, lm: LanguageModel, hmm_logLiklihood: pd.DataFrame):
        self.lm = lm
        self.hmm_logLiklihood = hmm_logLiklihood
        self.words = hmm_logLiklihood.columns
        self.num_inc_pattern = re.compile("(.*)\d+")
        self.BOS_ID = -1
        self.EOS_ID = -2
        self.BOS = "<s>"
        self.EOS = "</s>"

        self.NOT_FOUND_LM_PROB = -999

        # nodes
        self._layers = None
        self._parents = None
        self._childrens = None
        self._nodes_best_scores = None
        self._nodes_best_paths = None

    def recognize(self, ngram:int, sentence: list):
        # for viterbi algorithm include
        if ngram == 3:
            return self.recognize_tri(sentence)

    def recognize_tri(self, sentence: list):
        self.initialize_nodes(sentence)

        # forward
        self.forward(sentence)

        best_path = self.fetch_best_path()

        best_path_words =[self.to_word(word_id) for word_id in best_path[1:-1]]
        return best_path_words

    def initialize_nodes(self, sentence: list):
        word_ids = range(len(self.words))
        self._layers = [itertools.product(word_ids, word_ids) for _ in range(len(sentence))]


        self._childrens = [[
            [next_idx for next_idx, next_node in enumerate(self._layers[layer_id + 1])
             if node[1] == next_node[0]]
            for node in layer]
            for layer_id, layer in enumerate(self._layers)[:-1]]

        # for last node
        self._layers.append(zip(word_ids, [self.EOS] * len(word_ids)))

        self._parents = [[
            [prev_idx for prev_idx, prev_node in enumerate(self._layers[layer_id + 1])
             if node[0] == prev_node[1]]
            for node in layer]
            for layer_id, layer in enumerate(self._layers)[1:]]

        # for first node
        self._layers.insert(0, zip([self.BOS] * len(word_ids), word_ids))
        self._parents.insert(0, [[] for _ in self._layers[0]])

        # for last node
        self._childrens.append([[0] for _ in range(len(self._layers[-1]))])

    def forward(self, sentence: list):
        self._nodes_best_paths = []
        self._nodes_best_scores = []

        for layer_id, layer in enumerate(self._layers):
            if layer_id == 0:
                self._nodes_best_scores.append(
                    [self.get_logLiklihood(sentence[layer_id], node[1]) for node in layer])
                self._nodes_best_paths.append([])
                continue

            layer_best_scores = []
            layer_best_paths = []

            for node_idx, node in enumerate(layer):
                best_path, best_score = self.find_node_best_score(node_idx, node, layer_id, sentence[layer_id])
                layer_best_scores.append(best_score)
                layer_best_paths.append(best_path)

            self._nodes_best_scores.append(layer_best_scores)
            self._nodes_best_paths.append(layer_best_paths)

    def fetch_best_path(self):
        best_path = []

        node_idx = 0
        for layer_id in range(len(self._layers) - 1, 0, -1):
            prev_node_idx = self._nodes_best_paths[layer_id][node_idx]
            best_path.append(self._layers[layer_id - 1][prev_node_idx][1])
            node_idx = prev_node_idx

        best_path.append(self.BOS_ID)
        return list(reversed(best_path))

    def find_node_best_score(self, idx, node, layer_id, element_id):
        prev_node_idxs = self._parents[layer_id][idx]

        scores = [(prev_node_idx,
                   self.calc_score(node, self._layers[layer_id - 1][prev_node_idx])
                   + self._nodes_best_scores[layer_id - 1][prev_node_idx])
                  for prev_node_idx in prev_node_idxs]

        best_score = max(scores, key=lambda x: x[1])

        return best_score[0], best_score[1] + self.get_logLiklihood(element_id, node)

    def calc_score(self, node, prev_node):
        assert prev_node[1] == node[0]

        word_ids = (prev_node[0], node[0], node[1])
        ngram_prob = self.get_lm_word(3, word_ids)

        return ngram_prob

    def get_logLiklihood(self, element_id, word):
        if isinstance(word, int):
            return self.hmm_logLiklihood.loc[element_id, self.to_word(word)]

        return self.hmm_logLiklihood.loc[element_id, word]

    def to_word(self, word_id):
        if word_id == self.BOS_ID:
            return self.BOS
        elif word_id == self.EOS_ID:
            return self.EOS

        return self.words[word_id]

    def get_lm_word(self, ngram, word_ids):
        queries = [self.to_query(self.to_word(word_id)) for word_id in word_ids]

        prob = self.lm.get_prob(ngram, tuple(queries))

        # TODO require smoothing
        if prob is None:
            return self.NOT_FOUND_LM_PROB
        return np.log(np.power(prob, 10))

    def to_query(self, word):
        m = self.num_inc_pattern.match(word)
        if m:
            return m.group(1)
        return word

