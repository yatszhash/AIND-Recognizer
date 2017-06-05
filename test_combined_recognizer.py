import unittest

import math

from combined_recognizer import CombinedRecognizer


class TestCombinedRecognizer(unittest.TestCase):

    def test_initialize_possible_tokens(self):

        words = ["One", "Two"]
        sut = CombinedRecognizer(words, {})

        sut.initialize_possible_tokens(4)

        expected = [
            [(sut.BOS, "One", "Two"),
            (sut.BOS, "Two", "One"),
             (sut.BOS, "One", "One"),
             (sut.BOS, "Two", "Two")],
            [
             ("One", "One", "One"),
                ("One", "One", "Two"),
                ("One", "Two", "One"),
                ("One", "Two", "Two"),
                ("Two", "One", "One"),
                ("Two", "One", "Two"),
                ("Two", "Two", "One"),
                ("Two", "Two", "Two")
            ],
            [
                ("One", "One", "One"),
                ("One", "One", "Two"),
                ("One", "Two", "One"),
                ("One", "Two", "Two"),
                ("Two", "One", "One"),
                ("Two", "One", "Two"),
                ("Two", "Two", "One"),
                ("Two", "Two", "Two")
            ],
            [
                ("One", "Two", sut.EOS),
                ("Two", "One", sut.EOS),
                ("One", "One", sut.EOS),
                ("Two", "Two", sut.EOS)
            ]
        ]

        self.assertListEqual([len(p) for p in sut.possible_tokens],
                             [len(p) for p in expected],
                             "actual: {}\n\nexpected: {}".format(sut.possible_tokens,
                                                                 expected))
        for inner_actual, inner_expected in zip(sut.possible_tokens, expected):
            self.assertSetEqual(set(inner_actual), set(inner_expected),
                                "actual: {}\n\nexpected: {}".format(sut.possible_tokens,
                                               expected))

    def test_initialize_parents(self):
        words = ["One", "Two"]
        sut = CombinedRecognizer(words, {})

        sut.initialize_possible_tokens(4)

        sut.initialize_all_parents()

        self.assertListEqual(
            [len(layer) for layer in sut.parents],
            [4, 8, 8, 4]
        )

        self.assertListEqual(sut.parents[0][2], [])
        self.assertEqual(sut.possible_tokens[2][3],
                         ("One", "Two", "Two"))
        self.assertSetEqual(
            set(
                sut.get_parents_token(2, 3)
            ),
            set(
                [
                    ("One", "One", "Two"),
                    ("Two", "One", "Two")
                ]
            ),
            sut.parents[2][3]
        )

    def test_get_ngram_score(self):
        words = ["One", "Two"]

        ngram_liklihoods = {
             ("One", "One", "One"): -11,
        }

        sut = CombinedRecognizer(words, ngram_liklihoods)

        self.assertEqual(-11,
            sut.get_ngram_prob(("One", "One", "One")))

        self.assertEqual(CombinedRecognizer.NOT_EXIST_TOKEN_PROB,
                         sut.get_ngram_prob(("John", "Where", "Am")))

    def test_get_best_score_at(self):
        words = ["One", "Two"]

        ngram_liklihoods = {
            # begin
            (CombinedRecognizer.BOS, "One", "Two"): -1,
            (CombinedRecognizer.BOS, "Two", "One"): -2,
            (CombinedRecognizer.BOS, "One", "One"): -3,
            (CombinedRecognizer.BOS, "Two", "Two"): -4,

            # inner
            ("One", "One", "One"): -6,
            ("One", "One", "Two"): -7,
            ("One", "Two", "One"): 0,
            ("One", "Two", "Two"): -9,
            ("Two", "One", "One"): -10,
            ("Two", "One", "Two"): -2,
            ("Two", "Two", "One"): -14,
            ("Two", "Two", "Two"): -15,

            # end
            ("One", "One", CombinedRecognizer.EOS): -20,
            ("One", "Two", CombinedRecognizer.EOS): -21,
            ("Two", "One", CombinedRecognizer.EOS): -22,
            ("Two", "Two", CombinedRecognizer.EOS): -23
        }

        sut = CombinedRecognizer(words, ngram_liklihoods)

        sut.initialize_nodes(["a", "a", "a", "a"])

        sut.accumulate_best_scores()
        self.assertEqual(sut.get_best_score_at(0, 2), (None, -2))

        self.assertEqual(sut.possible_tokens[2][5], ("Two", "One", "Two"))

        # (BOS, One, Two, One, Two)
        self.assertEqual(sut.get_best_score_at(2, 5),
                         (2, -1 + 0 + -2))

        # (BOS, One, Two, One, Two, EOS)
        self.assertEqual(sut.get_best_score_at(2, 5),
                         (2, -1 + 0 + -2))

        self.assertEqual(sut.whole_best_score, -1 + 0 + -2 + -21)
        self.assertEqual(sut.whole_best_last_id_in_layer,
                         1)

    def test_fetch_best_sequence(self):
        words = ["1", "2", "3"]
        
        sut = CombinedRecognizer(words, {})

        sut.initialize_nodes(["a", "a", "a", "a", "a"])

        expected_words = ["BOS", "1", "2", "3", "2", "3", "EOS"]

        expected_path = [self.calc_id_in_layer_from_words(expected_words[id: id+3], 3)
                         for id, _ in enumerate(expected_words[:-2])]

        sut.best_parents = [[-999] * len(layer) for layer in sut.possible_tokens]

        assert len(expected_path) == len(sut.possible_tokens)
        sut.best_parents[0][expected_path[0]] = None
        for layer_id, id_in_layer in enumerate(expected_path[:-1]):
            sut.best_parents[layer_id + 1][expected_path[layer_id + 1]] = id_in_layer

        sut.whole_best_last_id_in_layer = expected_path[-1]

        self.assertEqual(sut.fetch_best_sequence(), expected_words[1:-1])

    def calc_id_in_layer_from_words(self, words, all_word_num):
        assert len(words) == 3

        if words[0] == "BOS":
            return (int(words[1]) - 1) * all_word_num + int(words[2]) - 1

        if words[-1] == "EOS":
            return (int(words[0]) - 1) * all_word_num + int(words[1]) - 1

        return (int(words[0]) - 1) * all_word_num * all_word_num \
               + (int(words[1]) - 1) * all_word_num \
               + int(words[2]) - 1
        

if __name__ == '__main__':
    unittest.main()
