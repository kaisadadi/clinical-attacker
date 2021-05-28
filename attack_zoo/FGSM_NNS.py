import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import json
import pickle
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer


class FGSMNNS():
    def __init__(self, config, model):
        #deepfool attack, deepfool + NNS
        self.model_path = r"/data/wke18/WORD_VEC/test_word2vec.txt"
        self.w2v_path = config.get("data", "w2v_path")
        self.word2id_path = config.get("data", "word2id_path")
        self.id2word_path = config.get("data", "id2word_path")
        #self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False, unicode_errors='ignore')
        self.model = model
        self.w2v = pickle.load(open(self.w2v_path, "rb"))
        self.word2id = json.load(open(self.word2id_path, "r"))
        self.id2word = json.load(open(self.id2word_path, "r"))
        self.max_len = config.getint("data", "max_seq_length")
        self.vec_size = config.getint("data", "vec_size")
        self.eps = config.getfloat("FGSMattack", "eps")
        self.max_epoch = config.getint("FGSMattack", "max_epoch")
        self.index_file = config.get("FGSMattack", "index_file")
        self.task = config.getint("FGSMattack", "task")
        if self.index_file == "None":
            self.indexer = AnnoyIndexer(model=self.model, num_trees=100)
            self.indexer.save(r"/data/wke18/data/ADV_EHR/ANNOY")
        else:
            self.indexer = AnnoyIndexer()
            self.indexer.load(self.index_file)

    def reverse_embedding(self, vec):
        vec = vec.detach().cpu().numpy()
        neighbor = self.model.most_similar([vec], topn=1, indexer=self.indexer)
        return neighbor

    def attack(self, model, dataset):
        for idx, data in enumerate(dataset):
            #prepare data
            for key in data.keys():
                if key == "input":
                    text = data[key].numpy()
                    bs = text.shape[0]
                    seq_len = text.shape[1]
                    embeded_text = np.zeros([bs, seq_len, self.vec_size])
                    for i in range(bs):
                        for j in range(seq_len):
                            embeded_text[i, j] = self.w2v[text[i, j]]
                    data[key] = Variable(torch.from_numpy(embeded_text)).cuda().float()
                if isinstance(data[key], torch.Tensor):
                    data[key] = Variable(data[key].cuda())
            #deep fool
            for batch in range(data["input"].shape[0]):
                true_label = data["label"][batch].detach().cpu().numpy()
                xadv = {"input": data["input"][batch], "label": data["label"][batch]}
                now_label = model(xadv, embedding=True)["prediction"].cpu().numpy()[0]
                print("start_label:", now_label)
                epoch_cnt = 0
                while np.sum(now_label != true_label) < self.task or epoch_cnt < self.max_epoch:
                    epoch_cnt += 1
                    loss = model(xadv, embedding=True)["loss"]
                    gradient = torch.autograd.grad(outputs=loss,
                                                   inputs=model.embeded,
                                                   grad_outputs=None,
                                                   retain_graph=True,
                                                   create_graph=False,
                                                   # allow_unused=True,
                                                   only_inputs=True)[0]
                    # print(gradient)
                    xadv["input"] = xadv["input"] + self.eps * torch.sign(gradient)
                    now_label = model(xadv, embedding=True)["prediction"].cpu().numpy()[0]
                print("epoch_cnt:", epoch_cnt)
                print("true_label:", true_label)
                print("now_label:", now_label)
                print()
                print(self.reverse_embedding(xadv["input"][0][0]))
                break
