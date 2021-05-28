import torch
import torch.autograd as autograd
import os
import pickle
import json
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

class DLNNS():
    def __init__(self, config):
        #deepfool attack, deepfool + NNS
        self.w2v_path = config.get("data", "w2v_path")
        self.word2id_path = config.get("data", "word2id_path")
        self.id2word_path = config.get("data", "id2word_path")
        self.w2v = pickle.load(open(self.w2v_path, "rb"))
        self.word2id = json.load(open(self.word2id_path, "r"))
        self.id2word = json.load(open(self.id2word_path, "r"))
        self.eta = config.getfloat("attack", "eta")
        self.eps = config.getfloat("attack", "eps")
        self.max_len = config.getint("data", "max_seq_length")
        self.max_epoch = config.getint("attack", "max_epoch")

    def attack(self, model, dataset):
        pass