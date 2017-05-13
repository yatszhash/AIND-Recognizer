import warnings

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = [get_predict_proba(models, test_set, item) for item in range(test_set.num_items)]

    guesses = [sort_and_extract(liklihood_list) for liklihood_list in probabilities]

    return probabilities, guesses


def get_predict_proba(models:dict, test_set:SinglesData, item:int):
    prob_words = {}

    for word, model in models.items():
        prob = None

        try:
            prob=model.score(
                *test_set.get_item_Xlengths(item))
        except:
            #print("failure on {} model with data No {}".format(word, item))
            pass

        prob_words[word] = prob

    return  prob_words

def sort_and_extract(liklihood_dict: dict):
    return max(filter(lambda key: liklihood_dict[key] is not None, liklihood_dict.keys()),
           key=lambda key: liklihood_dict[key])