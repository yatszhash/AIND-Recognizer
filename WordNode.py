import re
from collections import deque

import itertools
import numpy as np
import pandas as pd

from load_lm import LanguageModel


@DeprecationWarning
class WordNode(object):

    def __init__(self, ngram, word_ids, node_hmm_log_like, get_ngram_prob):
        if ngram == 3:
            assert len(word_ids) == 2
            self.word_ids = word_ids

            self.node_main_word_idx = 1

        elif ngram == 2:
            assert len(word_ids) == 1
            self.word_ids = word_ids

            self.node_main_word_idx = 0

        self.children = []
        self.parents = []

        self.layer_id = 0
        self.best_score = 0
        self.best_parent = None

        self.node_hmm_log_like = node_hmm_log_like

        self.get_ngram_prob = get_ngram_prob

    def find_best_parent(self):
        if not self.parents:
            assert self.best_score == 0
            return

        if len(self.parents) == 1:
            self.best_parent = self.parents[0]
            self.best_score = self.best_parent.best_score + self.node_hmm_log_like
            return

        ngram_Liklihoods = [(parent, self.calc_score(parent)) for parent in self.parents]

        best_score_parent = max(ngram_Liklihoods, key=lambda x: x[1])
        self.best_score = best_score_parent[1] + self.node_hmm_log_like
        self.best_parent = best_score_parent[0]

    def calc_score(self, parent_node):
        assert self.word_ids[0] == parent_node.word_ids[1]

        word_ids = (parent_node.word_ids[0], self.word_ids[0], self.word_ids[1])

        if None in word_ids or self.get_ngram_prob is None:
            return parent_node.best_score

        ngram_prob = self.get_ngram_prob(word_ids)
        return ngram_prob + parent_node.best_score


class HmmNgramRecognizer(object):
    def __init__(self, lm: LanguageModel, hmm_logLiklihood: pd.DataFrame):
        self.lm = lm
        self.hmm_logLiklihood = hmm_logLiklihood
        self.words = hmm_logLiklihood.columns
        self.all_word_ids = range(len(self.words))
        self.num_inc_pattern = re.compile("(.*)\d+")
        self.BOS_ID = -1
        self.EOS_ID = -2
        self.BOS = "<s>"
        self.EOS = "</s>"

        self._nodes = None
        self.NOT_FOUND_LM_PROB = -999

    def recognize(self, ngram: int, sentence: list):
        # for viterbi algorithm include
        if ngram == 3:
            return self.recognize_tri(sentence)

    def recognize_tri(self, sentence: list):
        self.initialize_nodes(sentence)

        # forward
        best_nodes = self.vitervi()

        del best_nodes[0]
        del best_nodes[-1]
        del best_nodes[-1]

        words = self.nodes_to_words_seq(best_nodes)

        return words

    def initialize_nodes(self, sentence):

        bos_node = WordNode(3, (None, self.BOS_ID), 0, None)
        bos_node.best_score = 0
        self._nodes = [[bos_node]]

        seq_len = len(sentence)

        get_ngram_prob = lambda word_ids: self.get_lm_word(3, word_ids)

        self._nodes.append([self.generate_node(1, sentence[0], (self.BOS_ID, word_id), get_ngram_prob)
                            for word_id in self.all_word_ids])

        for seq_idx in range(1, seq_len):
            self._nodes.append([self.generate_node(seq_idx + 1, sentence[seq_idx], tuple(word_ids), get_ngram_prob)
                            for word_ids in itertools.product(self.all_word_ids, self.all_word_ids)])

        self._nodes.append([self.generate_node(seq_len,
                                               sentence[seq_len - 1], (word_id, self.EOS_ID), get_ngram_prob)
                            for word_id in self.all_word_ids])

        eos_node = self.generate_node(seq_len + 2, None, (self.EOS_ID, None), None)
        eos_node.best_score = 0
        self._nodes.append([eos_node])

    def vitervi(self):

        for nodes_layer in self._nodes:
            for node in nodes_layer:
                node.find_best_parent()

        best_nodes = []


        parent = self._nodes[-1][0]
        while parent:
            best_nodes.append(parent)
            parent = parent.best_parent

        return list(reversed(best_nodes))

    def nodes_to_words_seq(self, nodes_path):
        return [self.to_word(node.word_ids[-1]) for node in nodes_path]

    def generate_node(self, layer_id, element_id, word_ids, get_ngram_prob):
        if word_ids[0] == self.EOS_ID:
            node = WordNode(3, word_ids, 0, get_ngram_prob)
        else:
            node = WordNode(3, word_ids,
                        self.get_logLiklihood(element_id, word_ids[1]),
                     get_ngram_prob)

        if layer_id == 0:
            return node

        node.parents = list(filter(lambda parent: parent.word_ids[1] == word_ids[0],
                                   self._nodes[layer_id - 1]))
        return node

    def get_logLiklihood(self, element_id, word):
        if isinstance(word, int):
            word_str = self.to_word(word)
            if word_str == "</s>":
                return 0

            return self.hmm_logLiklihood.loc[element_id, word_str]

        if word == "</s>":
            return 0

        return self.hmm_logLiklihood.loc[element_id, word]

    def get_lm_word(self, ngram, word_ids):
        queries = [self.to_query(self.to_word(word_id)) for word_id in word_ids]

        prob = self.lm.get_prob(ngram, tuple(queries))

        # TODO require smoothing
        if prob is None:
            return self.NOT_FOUND_LM_PROB
        return np.log(np.power(prob, 10))


    def to_word(self, word_id):
        if word_id == self.BOS_ID:
            return self.BOS
        elif word_id == self.EOS_ID:
            return self.EOS

        return self.words[word_id]

    def to_query(self, word):
        m = self.num_inc_pattern.match(word)
        if m:
            return m.group(1)
        return word