import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import numpy as np
import pickle
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

class ReverseEmbedding:
    def __init__(self, config, index_file=r"/data/wke18/data/ADV_EHR/ANNOY/index"):
        print("Loading ReverseEmbedding...")
        self.w2v_path = r"/data/wke18/WORD_VEC/test_word2vec.txt"
        self.w2v = KeyedVectors.load_word2vec_format(self.w2v_path, binary=False, unicode_errors='ignore')
        if index_file == None:
            self.indexer = AnnoyIndexer(model=self.w2v, num_trees=100)
            self.indexer.save(r"/data/wke18/data/ADV_EHR/ANNOY/index")
        else:
            self.indexer = AnnoyIndexer()
            self.indexer.load(index_file)

    def reverse_embedding(self, vec):
        print(vec)
        print(vec.shape)
        vec = vec.detach().cpu().numpy()
        neighbor = self.w2v.most_similar([vec], topn=1, indexer=self.indexer)
        print(neighbor)
        return neighbor



def show_important_words(model, dataset):
    #用于查看重要的词
    id2word = json.load(open(r"/data/wke18/data/ADV_EHR/DATA_50_V2/id2word.json"))
    for idx, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                    data[key] = Variable(data[key].cuda())
        loss = model(data)["loss"]
        print(loss.detach().cpu().numpy().tolist())
        gradient = torch.autograd.grad(outputs=loss,
                             inputs = model.embeded,
                             grad_outputs=None,
                             retain_graph=True,
                             create_graph=False,
                             only_inputs=True)[0].cpu().numpy()

        gradient = np.linalg.norm(gradient, axis=2, keepdims=False)
        gradient = np.argsort(-gradient, axis=1)

        for batch in range(data["input"].shape[0]):
            words = []
            for k in range(10):
                words.append(id2word[str(data["input"][batch][gradient[batch][k]].cpu().numpy())])
            print(words, data["label"][batch].cpu().numpy())
        break


def deepfool(model, dataset, eps=0.01):
    #eps是移动的距离
    pass

def fast_gradient(config, model, dataset, eps=0.01, sign=True, epochs = 5, clip_min=0., clip_max=1.):
    if sign:
        noise_fn = torch.sign
    else:
        noise_fn = lambda x:x
    re = ReverseEmbedding(config)
    glove = pickle.load(open(config.get("data", "w2v_path"), "rb"))
    for idx, data in enumerate(dataset):
        for key in data.keys():
            if key == "input":
                text = data[key].numpy()
                bs = text.shape[0]
                seq_len = text.shape[1]
                embeded_text = np.zeros([bs, seq_len, config.getint("data", "vec_size")])
                for i in range(bs):
                    for j in range(seq_len):
                        embeded_text[i, j] = glove[text[i, j]]
                data[key] = Variable(torch.from_numpy(embeded_text)).cuda().float()
            if isinstance(data[key], torch.Tensor):
                    data[key] = Variable(data[key].cuda())
        for batch in range(data["input"].shape[0]):
            true_label = data["label"][batch].detach().cpu().numpy()
            xadv = {"input": data["input"][batch], "label": data["label"][batch]}
            start_label = model(xadv, embedding = True)["prediction"].cpu().numpy()[0]
            now_label = start_label.copy()
            print("start_label:", start_label)
            epoch_cnt = 0
            while np.sum(now_label != true_label) < 0 or epoch_cnt < epochs:
                epoch_cnt += 1
                loss = model(xadv, embedding = True)["loss"]
                gradient = torch.autograd.grad(outputs=loss,
                                               inputs=model.embeded,
                                               grad_outputs=None,
                                               retain_graph=True,
                                               create_graph=False,
                                               only_inputs=True)[0]
                #print(gradient)
                xadv["input"] = xadv["input"] + eps * noise_fn(gradient)
                nowlabel = model(xadv, embedding = True)["prediction"].cpu().numpy()[0]
            print("epoch_cnt:", epoch_cnt)
            print("true_label:", true_label)
            print("now_label:", now_label)
            print(re.reverse_embedding(xadv["input"][0][0]))
            break

        break



