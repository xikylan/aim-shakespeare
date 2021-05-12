import collections
import math
import os
import re
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text


TOKENIZER = tf_text.UnicodeScriptTokenizer()


def predict(input):
    predictor = Predictor(source_vocabs='./source_vocabs.pickle',
                          target_vocabs='./target_vocabs.pickle',
                          tokenizer=TOKENIZER, model='./model_seq')

    output = predictor.predict(input)
    output = " ".join(x for x in output if x != '[UNK]')
    print(output)

    return output


class Predictor(object):
    # def __init__(self, path, source_vocabs, target_vocabs, tokenizer, model):
    def __init__(self, source_vocabs, target_vocabs, tokenizer, model):
        """
        Args:
          path : str
            Path to text data
          vocabs : str
            Path to dictionary of word : integer
          tokenizer : tensorflow_text.tokenizer object
          model : str
            Path to saved model
        """
        with open(source_vocabs, 'rb') as file:
            self.source_vocabs = pickle.load(file)
            self.source_vocabs = {x: i for x, i in zip(
                self.source_vocabs, range(len(self.source_vocabs)))}
        with open(target_vocabs, 'rb') as file:
            self.target_vocabs = pickle.load(file)
            self.target_vocabs = {i: x for x, i in zip(
                self.target_vocabs, range(len(self.target_vocabs)))}

        print('Dictionary loaded')

        self.tokenizer = tokenizer
        print('Tokenized loaded')
        self.model = tf.keras.models.load_model(
            model, custom_objects={'BLEU': BLEU})
        print('Model loaded')

    def process(self, input):
        output = input.lower()
        output = [x.decode('utf-8')
                  for x in self.tokenizer.tokenize(output).numpy()]

        output = ['[SOS]'] + output + ['[EOS]'] + \
            ['[PAD]']*(50 - 2 - len(output))

        output = [self.source_vocabs[x]
                  if x in self.source_vocabs else self.source_vocabs['[UNK]'] for x in output]
        return output

    def decode(self, input):
        # print(input.shape)
        return [self.target_vocabs[tok] for tok in input]

    def predict(self, input):
        # process input
        input = self.process(input)
        output = self.model.predict([input])[0]
        output = np.argmax(output, axis=-1)
        output = self.decode(output)
        return output


class BLEU(tf.keras.metrics.Metric):
    def __init__(self, name='BLEU', **kwargs):
        super(BLEU, self).__init__(name=name, **kwargs)
        self.bleu = self.add_weight(name='ctp', initializer='zeros')

    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.
        Returns:
          The Counter containing all n-grams upto max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def _compute_bleu_score(self, x, y,
                            ratios, smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
          reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
          translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
          max_order: Maximum n-gram order to use when computing BLEU score.
          smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
          3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
          precisions and brevity penalty.
        """

        def _func(reference_corpus, translation_corpus):
            reference_corpus = reference_corpus.numpy()
            translation_corpus = translation_corpus.numpy()

            matches_by_order = [0] * len(ratios)
            possible_matches_by_order = [0] * len(ratios)
            reference_length = 0
            translation_length = 0
            for (reference, translation) in zip(reference_corpus,
                                                translation_corpus):
                reference_length += len(reference)
                translation_length += len(translation)
                #tf.print(len(reference), len(translation))
                merged_ref_ngram_counts = collections.Counter()

                merged_ref_ngram_counts |= self._get_ngrams(
                    reference, len(ratios))
                translation_ngram_counts = self._get_ngrams(
                    translation, len(ratios))
                overlap = translation_ngram_counts & merged_ref_ngram_counts
                for ngram in overlap:
                    matches_by_order[len(ngram)-1] += overlap[ngram]
                for order in range(1, len(ratios)+1):
                    possible_matches = len(translation) - order + 1
                    if possible_matches > 0:
                        possible_matches_by_order[order-1] += possible_matches

            precisions = [0] * len(ratios)
            for i in range(0, len(ratios)):
                if smooth:
                    precisions[i] = ((matches_by_order[i] + 1.) /
                                     (possible_matches_by_order[i] + 1.))
                else:
                    if possible_matches_by_order[i] > 0:
                        precisions[i] = (float(matches_by_order[i]) /
                                         possible_matches_by_order[i])
                    else:
                        precisions[i] = 0.0

            if min(precisions) > 0:
                p_log_sum = sum(r * math.log(p)
                                for p, r in zip(precisions, ratios))
                geo_mean = math.exp(p_log_sum)
            else:
                geo_mean = 0

            ratio = float(translation_length) / reference_length

            if ratio > 1.0:
                bp = 1.
            else:
                bp = math.exp(1 - 1. / ratio)

            bleu = geo_mean * bp
            #tf.print('bleu', bleu)

            return bleu

        bleu = tf.py_function(func=_func, inp=[x, y], Tout=tf.float32)
        return bleu

    def update_state(self, y_true, y_pred, sample_weight=None):
        bleu = self._compute_bleu_score(y_true, y_pred, smooth=True,
                                        ratios=[0.15, 0.3, 0.3, 0.25])
        self.bleu.assign(bleu)

    def result(self):
        return self.bleu

    def reset_states(self):
        self.bleu.assign(0)


if __name__ == "__main__":
    predict()
