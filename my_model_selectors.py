import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold

import asl_utils
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores


        best_model = None
        best_score = float('-inf')

        hyper_paramaters = range(self.min_n_components, self.max_n_components + 1)


        for n_components in hyper_paramaters:
            try:
                current_model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)

                l = current_model.score(self.X, self.lengths)
                #current_score = self.bic(l,  ,len(self.X))
                current_score = 0

                if current_score > best_score:
                    best_model = current_model
                    best_score = current_score
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_components))

        return best_model

    @staticmethod
    def bic(l, p, n):
        return -2 * np.log(l) + p * np.log(n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float('-inf')

        hyper_paramaters = range(self.min_n_components, self.max_n_components + 1)

        n_split = 3
        if n_split > len(self.sequences):
            n_split = len(self.sequences)
        kfold = KFold(n_split, random_state=self.random_state)
        train, test = next(kfold.split(self.sequences))

        train_X, train_lengths = asl_utils.combine_sequences(train, self.sequences)
        test_X, test_lengths = asl_utils.combine_sequences(test, self.sequences)

        for n_components in hyper_paramaters:
            try:
                current_model = GaussianHMM(n_components=n_components, n_iter=1000)\
                    .fit(train_X, train_lengths)

                current_score = current_model.score(test_X, test_lengths)

                if current_score > best_score:
                    best_model = current_model
                    best_score = current_score
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_components))

        return best_model
