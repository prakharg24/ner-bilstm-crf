import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

torch.manual_seed(1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, word_to_ix, char_to_ix):
    idxs = []
    caps = []
    # lngth = []
    for w in seq:
        if w[0].isupper():
            caps.append(1.0)
        else:
            caps.append(0.0)
        if w in word_to_ix:
            idxs.append(word_to_ix[w])
        else:
            idxs.append(word_to_ix['Unk'])
    tensor = torch.LongTensor(idxs)
    fidxs = []
    for w in seq:
        cidxs = []
        for c in w:
            if c in char_to_ix:
                cidxs.append(char_to_ix[c])
            else:
                cidxs.append(char_to_ix['#'])
        fidxs.append(autograd.Variable(torch.LongTensor(cidxs)))
    return autograd.Variable(tensor), fidxs, autograd.Variable(torch.FloatTensor(caps).view(-1, 1))


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, char_size, tag_to_ix, embedding_dim, char_embedding, hidden_dim, hidden_dim_char):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.char_embedding = char_embedding
        self.hidden_dim = hidden_dim
        self.hidden_dim_char = hidden_dim_char
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeds = nn.Embedding(char_size, char_embedding)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.char_lstm = nn.LSTM(char_embedding, hidden_dim_char // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2tag_char = nn.Linear(hidden_dim_char, self.tagset_size)

        self.jointhem = nn.Linear(2*self.tagset_size + 1, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
        self.char_hidden = self.init_char_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def init_char_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim_char // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim_char // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_char_lstm(self, word):
        self.char_hidden = self.init_char_hidden()
        old_embeds = self.char_embeds(word)
        embeds = old_embeds.view(len(word), 1, -1)
        lstm_out, self.char_hidden = self.char_lstm(embeds, self.char_hidden)
        lstm_out = lstm_out.view(len(word), self.hidden_dim_char)
        lstm_feats = self.hidden2tag_char(lstm_out)
        return lstm_feats[-1].view(1, -1)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, words, caps, tags):
        feats = self._get_lstm_features(sentence)

        final_char = self._get_char_lstm(words[0])
        for word in words[1:]:
            lstm_char_feats = self._get_char_lstm(word)
            final_char = torch.cat([final_char, lstm_char_feats])

        feats = torch.cat([feats, final_char], dim=1)
        feats = torch.cat([feats, caps], dim=1)
        feats = self.jointhem(feats)

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, words, caps):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        final_char = self._get_char_lstm(words[0])
        for word in words[1:]:
            lstm_char_feats = self._get_char_lstm(word)
            final_char = torch.cat([final_char, lstm_char_feats])

        lstm_feats = torch.cat([lstm_feats, final_char], dim=1)
        lstm_feats = torch.cat([lstm_feats, caps], dim=1)
        lstm_feats = self.jointhem(lstm_feats)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
CHAR_DIM = 30
HIDDEN_DIM = 50
HIDDEN_DIM_CHAR = 30

lines = [line.rstrip('\n') for line in open('/home/cse/btech/cs1150245/scratch/train.txt')]

full_data = []

word1 = []
tag = []
for word in lines:
    curr_word = word.split()
    if not curr_word:
        full_data.append((word1,tag))
        word1 = []
        tag = []
    else:
        word1.append(curr_word[0])
        tag.append(curr_word[1])

print(len(full_data))

training_data = full_data[:3000]
test_data = full_data[3000:]

max_word_len = 0

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if(len(word)>max_word_len):
            max_word_len = len(word)
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

char_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

word_to_ix['Unk'] = len(word_to_ix)
if not '#' in char_to_ix:
    char_to_ix['#'] = len(char_to_ix)

tag_to_ix = {"O": 0, "T": 1, "D": 2, START_TAG: 3, STOP_TAG: 4}

print(len(char_to_ix))

model = BiLSTM_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, EMBEDDING_DIM, CHAR_DIM, HIDDEN_DIM, HIDDEN_DIM_CHAR)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
# print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in, words_in, caps_in = prepare_sequence(sentence, word_to_ix, char_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, words_in, caps_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()
    # torch.save(model.state_dict(), '/home/cse/btech/cs1150245/scratch/model' + '_' + str(epoch) + '.pth')
    corr_arr = []
    pred_arr = []
    for sent in test_data:
        precheck_sent, precheck_words, precheck_caps = prepare_sequence(sent[0], word_to_ix, char_to_ix)
        some_model = model(precheck_sent, precheck_words, precheck_caps)
        ans_tag = some_model[1]
        for corr_tag, pred in zip(sent[1], ans_tag):
            corr_arr.append(tag_to_ix[corr_tag])
            pred_arr.append(pred)
    print(metrics.f1_score(corr_arr, pred_arr, average='macro', labels=[1, 2]))

# def getscore()