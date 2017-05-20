import re


class LanguageModel:

    def __init__(self):
        self.ngrams = {}
        self.backoffs = {}

    def get_prob(self, ngram: int, words: tuple):
        if ngram != len(words):
            raise ValueError("n of ngram and words length of words must be equal")

        if ngram not in self.ngrams:
            raise ValueError("{} gram model does not exist".format(ngram))

        if words not in self.ngrams[ngram]:
            return None

        return self.ngrams[ngram][words]

    def load_from_file(self, file_path: str):
        '''
        load from apla file
        :return: 
        '''

        pattern_header_sec = re.compile(r"\n\\data\\\n(.+?)\n\\", flags=re.DOTALL)
        pattern_ngram_length = re.compile(r"^ngram\s+\d+=(\d+)$", flags=re.MULTILINE)

        pattern_bodies = re.compile(r"\\\d+-grams:\n(.+?)\n\n", flags=re.DOTALL)

        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

            header = pattern_header_sec.match(full_text)
            if not header:
                raise ValueError("The header is not the correct format.")
            ngram_length = list(map(lambda x: int(x), pattern_ngram_length.findall(header.group(1))))

            if not len(ngram_length):
                raise ValueError("The header is not the correct format.")

            bodies = pattern_bodies.findall(full_text)

            if not len(bodies):
                raise ValueError("The body is not the correct format.")

            if len(bodies) != len(ngram_length):
                raise ValueError("The numbers of the items :{} in the bodies and "
                                 "the number of the items in the header aren't consistent"
                                 .format(len(bodies), len(ngram_length)))

            for index, body in enumerate(bodies):
                prob_words = [re.split(r"[\t\s]", line) for line in body.split("\n")]

                if not len(bodies):
                    raise ValueError("The body is not the correct format.")

                if len(prob_words) != ngram_length[index]:
                    raise ValueError("The numbers of the items in the bodies and "
                                     "the number of the items in the header aren't consistent")
                ngram = index + 1
                self.ngrams[ngram] = {}

                for prob_word in prob_words:
                    len_prob_word = len(prob_word)
                    if len_prob_word < ngram + 1 or len_prob_word > ngram + 2:
                        raise ValueError("The body is not the correct format.")

                    self.ngrams[ngram][tuple(prob_word[1:1 + ngram])] = float(prob_word[0])

                    if len_prob_word == ngram + 2 and ngram not in self.backoffs:
                        self.backoffs[ngram]  = {}

                    if len_prob_word == ngram + 2:
                        self.backoffs[ngram][tuple(prob_word[1:1 + ngram])] = float(prob_word[-1])


if __name__ == "__main__":
    # for debug

    lm = LanguageModel()
    file_path = "../devel-lm-M3.sri.lm"

    lm.load_from_file(file_path)
    print(lm.ngrams)