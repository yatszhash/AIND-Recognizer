import unittest

from combined_recognizer import CombinedRecognizer


class TestCombinedRecognizer(unittest.TestCase):

    def test_initialize_possible_tokens(self):

        words = ["One", "Two"]
        sut = CombinedRecognizer(words)

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

        sut.initialize_parents()

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
            ("One", "Two", CombinedRecognizer.EOS): -20,
            ("Two", "One", CombinedRecognizer.EOS): -21,
            ("One", "One", CombinedRecognizer.EOS): -22,
            ("Two", "Two", CombinedRecognizer.EOS): -23
        }

        sut = CombinedRecognizer(words, ngram_liklihoods)
        sut.initialize_possible_tokens(4)

        self.assertEqual(sut.get_best_score_at(0, 2), -2)
        self.assertEqual(sut.get_best_score_at(2, 5),
                         -3 + 0 + -2)

if __name__ == '__main__':
    unittest.main()
