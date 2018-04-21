import nltk
import sklearn_crfsuite
from sklearn import metrics
from sklearn_crfsuite import metrics
import numpy as np
import re
import pickle
import sys

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'position': i,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-2:]_rep': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'len(word)': len(word),
        'word.isupper()': word.isupper(),
        'word[0].isupper()': word[0].isupper(),
        'postag': postag,
    }
    for w, x, y, z in zip(word, word[1:], word[2:], word[3:]):
        quad = w + x + y + z
        if quad in features:
            features.update({
                quad:(features[quad]+1),
            })
        else:
            features.update({
                quad:1,
            })
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word[0].isupper()': word1[0].isupper(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word[0].isupper()': word1[0].isupper(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    for j in range(0, len(sent)):
        new_str = str(i - j)
        features[new_str + ":" + sent[j][0]] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag in sent]

def sent2tokens(sent):
    return [token for token, postag in sent]

lines = [line.rstrip('\n') for line in open(sys.argv[1])]

tagless_data = []

word1 = []
for word in lines:
    curr_word = word.split()
    if not curr_word:
        tagless_data.append(word1)
        word1 = []
    else:
        word1.append(curr_word[0])

tagless_data.append(word1)

full_data = []

for sent in tagless_data:
    just_sent = []
    for word in sent:
        just_sent.append(word)
    sent_tag = nltk.pos_tag(just_sent)
    full_data.append(sent_tag)


X_test = [sent2features(s) for s in full_data]

with open('crf_model.pkl', 'rb') as file:
    crf = pickle.load(file)

my_pred = crf.predict(X_test)

wr_file = open(sys.argv[2], 'w')

ite = 0

for sent, tag_sent in zip(full_data, my_pred):
    if(ite!=0):
        wr_file.write("\n")        
    for word, tag in zip(sent, tag_sent):
        wr_file.write(word[0] + " " + tag)
        wr_file.write("\n")
    ite += 1

wr_file.close()