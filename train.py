import nltk
import sklearn_crfsuite
from sklearn import metrics
from sklearn_crfsuite import metrics
import numpy as np
import re
import pickle

lines = [line.rstrip('\n') for line in open('train.txt')]

tagless_data = []

word1 = []
for word in lines:
    curr_word = word.split()
    if not curr_word:
        tagless_data.append(word1)
        word1 = []
    else: 
        word1.append((curr_word[0], curr_word[1]))

full_data = []

for sent in tagless_data:
    just_sent = []
    for word in sent:
        just_sent.append(word[0])
    sent_tag = nltk.pos_tag(just_sent)
    temp_sent = []
    for word, word_tag in zip(sent, sent_tag):
        temp_sent.append((word_tag[0], word_tag[1], word[1]))
    full_data.append(temp_sent)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
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
    return [label for token, postag, label in sent]

np.random.shuffle(full_data)

X_train = [sent2features(s) for s in full_data]
y_train = [sent2labels(s) for s in full_data]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, 
    c2=0.1, 
    max_iterations=100,
    all_possible_transitions=False,
)
crf.fit(X_train, y_train)

my_pred = crf.predict(X_train)
print(metrics.flat_f1_score(my_pred, y_train, average='macro', labels=['D', 'T']))

with open('crf_model.pkl', 'wb') as file:
    pickle.dump(crf, file)