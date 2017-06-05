
from itertools import product
import numpy as np

class CombinedRecognizer(object):
    BOS = "<s>"
    EOS = "</s>"
    NOT_EXIST_TOKEN_PROB = -999

    def __init__(self, words, ngram_probs):
        self.ngram = 3

        self.possible_tokens = None
        self.words = words
        self.ngram_probs = ngram_probs

        self.best_scores = None
        self.best_parents = None
        self.parents = None

    def initialize_nodes(self, seq: list):
        self.initialize_possible_tokens(len(seq))
        self.initialize_all_parents()

    def initialize_possible_tokens(self, seq_len: int):
        self.possible_tokens = [self.generate_first_layer()]

        inner_len = seq_len - self.ngram + 1
        self.possible_tokens.extend(
            [self.generate_inner_layer() for _ in range(inner_len)]
        )

        self.possible_tokens.append(
            self.generate_last_layer()
        )

    def generate_first_layer(self):
        return [(self.BOS, word1, word2) for word1, word2
                in product(self.words, repeat=2)]

    def generate_last_layer(self):
        return [(word1, word2, self.EOS)
            for word1, word2 in product(self.words, repeat=2)]

    def generate_inner_layer(self):
        return list(product(self.words, repeat=3))

    def initialize_all_parents(self):
        self.parents = [
            [self.extract_parent(layer_id, token)
             for id_in_layer, token in enumerate(layer)]
            for layer_id, layer in enumerate(self.possible_tokens)
        ]

    def get_parents_token(self, layer_id, id_in_layer):
        return [
            self.possible_tokens[layer_id - 1][parent_id_in_layer]
            for parent_id_in_layer in self.parents[layer_id][id_in_layer]
        ]

    def get_parents_best_score(self, layer_id, id_in_layer):
        return [
            self.best_scores[layer_id - 1][parent_id_in_layer]
            for parent_id_in_layer in self.parents[layer_id][id_in_layer]
        ]

    def extract_parent(self, layer_id, token):
        if layer_id == 0:
            return []

        return [prev_token_id_in_layer
                for prev_token_id_in_layer, prev_token
            in enumerate(self.possible_tokens[layer_id - 1])
            if prev_token[1] == token[0] and prev_token[2] == token[1]
        ]

    def get_ngram_prob(self, token):
        if not token in self.ngram_probs:
            return self.NOT_EXIST_TOKEN_PROB
        return self.ngram_probs[token]

    @DeprecationWarning
    def initialize_best_scores(self):
        self.best_scores = [[float("-inf")] * len(layer)
                            for layer in self.possible_tokens]

    def accumulate_best_scores(self):
        self.best_scores = []
        self.best_parents = [[float("-inf")] * len(layer) for layer in self.possible_tokens]
        for layer_id, layer in enumerate(self.possible_tokens):
                self.best_scores.append(
                [
                    self.find_accumulate_best_score(layer_id, id_in_layer)
                    for id_in_layer, _ in enumerate(layer)
                ])

    def find_accumulate_best_score(self, layer_id, id_in_layer):
        assert len(self.best_scores) == layer_id
        parents_scores = self.get_parents_best_score(layer_id, id_in_layer)

        current_score = self.get_ngram_prob(self.possible_tokens[layer_id][id_in_layer])
        if len(parents_scores) == 0:
            self.best_parents[layer_id][id_in_layer] = None
            return current_score

        best_parent_score = max(parents_scores)
        self.best_parents[layer_id][id_in_layer] \
            = self.parents[layer_id][id_in_layer][np.argmax(parents_scores)]

        return current_score + best_parent_score

    def get_best_score_at(self, layer_id, id_in_layer):
        return self.best_parents[layer_id][id_in_layer], self.best_scores[layer_id][id_in_layer]
