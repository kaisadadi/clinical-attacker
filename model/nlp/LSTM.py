import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tools.accuracy_init import init_accuracy_function
from model.loss import multi_label_cross_entropy_loss


class LSTM(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LSTM, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.w2v = np.load(config.get("data", "w2v_path"))
        self.embedding = nn.Embedding.from_pretrained(
                                    embeddings=torch.from_numpy(self.w2v),
                                    freeze = True
                                    )
        self.LSTM = nn.LSTM(input_size = 200,
                            hidden_size = 512,
                            num_layers = 2,
                            bias = True,
                            batch_first = True,
                            bidirectional = True)

        #self.fc = nn.Linear(4 * 512, self.output_dim)
        self.fc = nn.Sequential(
            nn.Linear(4 * 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.output_dim),
        )

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
        else:
            self.embeded = x.requires_grad_(True)

        _, (y, _) = self.LSTM(self.embeded)
        y = y.permute(1, 0, 2)
        y = y.reshape(y.size()[0], -1)
        y = self.fc(y)
        y = y.view(y.size()[0], -1)
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
