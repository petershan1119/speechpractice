import re

import nltk

from nltk.corpus import gutenberg
from string import punctuation

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

bible = gutenberg.sents('bible-kjv.txt')
remove_terms = punctuation + '0123456789'

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc




norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
