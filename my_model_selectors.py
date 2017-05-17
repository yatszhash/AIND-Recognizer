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

        best_model = None
        best_score = float('inf')

        hyper_parameters = range(self.min_n_components, self.max_n_components + 1)


        for n_components in hyper_parameters:
            try:
                current_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                l = current_model.score(self.X, self.lengths)

                current_score = self.bic(l, n_components, len(self.X[0]) ,len(self.lengths))

                if current_score < best_score:
                    best_model = current_model
                    best_score = current_score

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_components))

        return best_model

    def bic(self, l, n_states, n_features, n_data):

        # number of transition probabilities
        n_transition = n_states * (n_states - 1)

        # number of start state
        n_start = n_states - 1

        # number of mean and diagonal covariance for emission probabilities
        n_emission = 2 * n_states * n_features

        n_params = n_transition + n_start +  n_emission

        return -2 * l+ (n_params)* np.log(n_data)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        best_model = None
        best_score = float('-inf')

        hyper_parameters = range(self.min_n_components, self.max_n_components + 1)


        for n_components in hyper_parameters:
            try:
                current_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                current_score = self.dic(current_model)

                if current_score > best_score:
                    best_score = current_score
                    best_model = current_model

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_components))


        return best_model

    def dic(self, model):
        l = model.score(self.X, self.lengths)

        all_scores = [model.score(self.hwords[word][0], self.hwords[word][1])
                      for word in self.words if word != self.this_word]

        anti_l = sum(all_scores) \
                 / (len(self.words) - 1)

        return l - anti_l

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float('-inf')
        best_n_comp = 0

        hyper_paramaters = range(self.min_n_components, self.max_n_components + 1)

        n_split = 3

        for n_components in hyper_paramaters:
            try:
                if n_split > len(self.sequences):
                    current_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False) \
                                    .fit(self.X, self.lengths)
                    current_score = current_model.score(self.X, self.lengths)

                    if current_score > best_score:
                        best_model = current_model
                        best_score = current_score

                else:
                    kfold = KFold(n_split, random_state=self.random_state)

                    temp_scores = []

                    for train, test in kfold.split(self.sequences):
                        train_X, train_lengths = asl_utils.combine_sequences(train, self.sequences)
                        test_X, test_lengths = asl_utils.combine_sequences(test, self.sequences)

                        temp_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)\
                        .fit(train_X, train_lengths)

                        temp_scores.append(temp_model.score(test_X, test_lengths))

                    current_score = np.average(temp_scores)

                    if current_score > best_score:
                        best_n_comp = n_components
                        best_score = current_score

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n_components))

        if n_split <= len(self.sequences) and best_n_comp > 0:
            try:
                best_model = GaussianHMM(n_components=best_n_comp, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)\
                        .fit(self.X, self.lengths)

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, best_n_comp))

        return best_model
