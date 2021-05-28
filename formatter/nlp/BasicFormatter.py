import json
import torch
import numpy as np

from formatter.Basic import BasicFormatter


class BasicLSTMFormatter(BasicFormatter):
    #分词，word2id，label，UNK和PAD的处理
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.word2id_path = config.get("data", "word2id_path")
        self.word2id = json.load(open(self.word2id_path, "r"))
        self.max_len = config.getint("data", "max_seq_length")
        self.output_dim = config.getint("model", "output_dim")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input = []
        label = []

        for item in data:
            TEXT = item["text"]
            token = []
            for word in TEXT:
                word = word.lower()
                if word in self.word2id.keys():
                    token.append(self.word2id[word])
                else:
                    token.append(1)  #UNK 1

            while len(token) < self.max_len:
                token.append(0)
            token = token[0:self.max_len]

            input.append(token)
            item_label = [0] * self.output_dim
            for disease in item["label"]:
                item_label[disease] = 1

            label.append(item_label)

        input = torch.from_numpy(np.array(input)).long()

        label = torch.from_numpy(np.array(label)).long()

        return {'input': input, 'label': label}
