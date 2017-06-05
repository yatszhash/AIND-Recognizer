from unittest import TestCase
import  unittest

import pandas as pd
import numpy as np

from WordNode import HmmNgramRecognizer
from asl_data import AslDb
from hmm_ngram import OldHmmNgramRecognizer
from load_lm import LanguageModel

@DeprecationWarning
class TestHmmNgramRecognizer(TestCase):

    def test_recognize_tri(self):
        asl = AslDb()

        features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
        asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
        asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']


        asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
        asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

        features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

        polar_r = lambda row: np.sqrt(row[0] ** 2 + row[1] ** 2)
        polar_theta = lambda row: np.arctan2(row[0], row[1])

        asl.df[features_polar[0]] = asl.df[['grnd-rx', 'grnd-ry']].apply(polar_r, axis=1)
        asl.df[features_polar[1]] = asl.df[['grnd-rx', 'grnd-ry']].apply(polar_theta, axis=1)

        asl.df[features_polar[2]] = asl.df[['grnd-lx', 'grnd-ly']].apply(polar_r, axis=1)
        asl.df[features_polar[3]] = asl.df[['grnd-lx', 'grnd-ly']].apply(polar_theta, axis=1)

        test_set = asl.build_test(features_polar)
        test_hmm_model_file = "data/hmm_result_polar_dic.csv"

        df_probs = pd.read_csv(test_hmm_model_file)
        del df_probs["Unnamed: 0"]

        file_path = "data/devel-lm-M3.sri.lm"
        lm = LanguageModel()
        lm.load_from_file(file_path)

        sut = HmmNgramRecognizer(hmm_logLiklihood=df_probs, lm=lm)

        guess = sut.recognize_tri(test_set.sentences_index[2])

        print(guess)

if __name__ == "__main__":
    unittest.main()