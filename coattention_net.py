import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn


class CoattentionNet(nn.Module):

    def __init__(self, num_embeddings, num_classes, embed_dim=512, k=30):
        super().__init__()
        # Word level embeddings
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        # Phrase level embeddings
        self.unigram_conv = nn.Conv1d(embed_dim, embed_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(embed_dim, embed_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        # Question level embeddings
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=3, dropout=0.4)
        self.tanh = nn.Tanh()

        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, image, question):
        question, lens = rnn.pad_packed_sequence(question)
        question = question.permute(1, 0)
        words = self.embed(question).permute(0, 2, 1)

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2)
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2)
        words = words.permute(0, 2, 1)

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1)

        hidden = None
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)
        # Attention on word level features
        v_word, q_word = self.parallel_coattention(image, words)
        # Attention on phrase level features
        v_phrase, q_phrase = self.parallel_coattention(image, phrase)
        # Attention on question level features
        v_sent, q_sent = self.parallel_coattention(image, sentence)

        h_w = self.tanh(self.W_w(q_word + v_word))
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))

        probs = self.fc(h_s)

        return probs

    def parallel_coattention(self, V, Q):
        # Computing affinity matrix
        C = torch.matmul(Q, torch.matmul(self.W_b, V))

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))

        # Attention weights for image
        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # Attention weights for question
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        q = torch.squeeze(torch.matmul(a_q, Q))

        return v, q
