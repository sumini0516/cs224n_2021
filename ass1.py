import sys

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]
import nltk

nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)

'''
Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, 
translation, etc., so it is important to build some intuitions as to their strengths and weaknesses. Here, you will explore two types of word vectors: those derived from *co-occurrence matrices*, and those derived via *GloVe*. 

**Note on Terminology:** The terms "word vectors" and "word embeddings" are often used interchangeably. 
The term "embedding" refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. 
As [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding) states, "*conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension*".
'''

'''
Many word vector implementations are driven by the idea that similar words, i.e., (near) synonyms, will be used in similar contexts. 
As a result, similar words will often be spoken or written along with a shared subset of words, i.e., contexts. 
By examining these contexts, we can try to develop embeddings for our words. 
With this intuition in mind, many “old school” approaches to constructing word vectors relied on word counts. 
Here we elaborate upon one of those strategies, co-occurrence matrices.
'''


def read_corpus(category="crude"):
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
            Params:
                corpus (list of list of strings): corpus of documents
            Return:
                corpus_words (list of strings): sorted list of distinct words across the corpus
                num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    for sentence in corpus:
        for word in sentence:
            if word not in corpus_words:
                corpus_words.append(word)
            else:
                continue
    corpus_words = sorted(corpus_words)
    num_corpus_words = len(corpus_words)
    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    M = None
    word2ind = {}
    test_corpus_words, test_num_corpus_words = distinct_words(corpus)
    # print(test_corpus_words)
    M = np.zeros((test_num_corpus_words, test_num_corpus_words))
    for word in test_corpus_words:
        word2ind[word] = test_corpus_words.index(word)
    # print(word2ind)
    # print(corpus)
    for sentence in corpus:
        for idx, word_id in enumerate(sentence):
            # print(idx, word_id)
            word_index = word2ind[word_id] #단어의 index -> well:9
            # print(word_index)
            for size in range(1, window_size + 1):  # window_size=1 -> range(1,2)
                left_idx = idx - size  # 2-1=1
                right_idx = idx + size  # 2+1=3
                if left_idx >= 0:
                    left_word_id = sentence[left_idx]
                    left_word_index = word2ind[left_word_id]
                    M[word_index, left_word_index] += 1
                if right_idx < len(sentence):
                    right_word_id = sentence[right_idx]
                    right_word_index = word2ind[right_word_id]
                    M[word_index, right_word_index] += 1
    # print(M)
    # print(word2ind)
    return M, word2ind


'''
# Question 1.1
reuters_corpus = read_corpus()
# pprint.pprint(reuters_corpus[:3], compact=True, width=100)

# Define toy corpus
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "),
               "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)
print(test_corpus_words, num_corpus_words)
'''
'''
# Correct answers
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
'''

'''
# Question 1.2
# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "),
               "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2ind
M_test_ans = np.array(
    [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
     [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., ],
     [0., 1., 0., 0., 0., 0., 0., 0., 1., 0., ],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., ],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., ],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., ],
     [0., 0., 1., 0., 1., 1., 0., 0., 0., 1., ],
     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0., ]]
)
print(M_test_ans)
ans_test_corpus_words = sorted(
    [START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
word2ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))
print(word2ind_ans)
# Test correct word2ind
assert (word2ind_ans == word2ind_test), "Your word2ind is incorrect:\nCorrect: {}\nYours: {}".format(word2ind_ans, word2ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2ind_ans.keys():
    idx1 = word2ind_ans[w1]
    for w2 in word2ind_ans.keys():
        idx2 = word2ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)
'''