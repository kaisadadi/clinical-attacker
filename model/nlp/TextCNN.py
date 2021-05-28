import torch
import torch.nn as nn
from tools.accuracy_init import init_accuracy_function
from model.loss import multi_label_cross_entropy_loss
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TextCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.w2v = np.load(config.get("data", "w2v_path"))
        self.embedding = nn.Embedding.from_pretrained(
                                    embeddings=torch.from_numpy(self.w2v),
                                    freeze = True
                                    )

        self.min_gram = config.getint("model", "min_gram")
        self.max_gram = config.getint("model", "max_gram")
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, self.output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.criterion = multi_label_cross_entropy_loss
        self.accuracy_function = init_accuracy_function(config, *args, **params)
        self.sigmoid = nn.Sigmoid()
        self.embeded = None


    def forward(self, data, config=None, gpu_list=None, acc_result=None, mode=None, embedding=False):
        x = data['input']
        if len(x.shape) == 1 and embedding == False:
            x = x.unsqueeze(0)
        if len(x.shape) == 2 and embedding == True:
            x = x.unsqueeze(0)


        if embedding == False:
            self.embeded = self.embedding.forward(x).requires_grad_(True)
            x = self.embeded.view(self.embeded.shape[0], 1, -1, self.data_size)
        else:
            self.embeded = x.requires_grad_(True)
            x = self.embeded.view(self.embeded.shape[0], 1, -1, self.data_size)

        conv_out = []
        gram = self.min_gram

        for conv in self.convs:
            try:
                y = self.relu(conv(x))
            except:
                print(data["input"])
                print(data["input"].shape)
                print(x.shape)
                print(conv(x).shape)
                gg
            y = torch.max(y, dim=2)[0].view(x.shape[0], -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        y = self.fc(self.dropout(conv_out))
        y = self.sigmoid(y)

        logits = y
        prediction = torch.ge(y, 0.5)

        if "label" in data.keys():
            label = data["label"]
            if len(label.shape) == 1:
                label = label.unsqueeze(0)
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result, "prediction": prediction, "logits": logits}

        return {"logits": logits, "prediction": prediction}