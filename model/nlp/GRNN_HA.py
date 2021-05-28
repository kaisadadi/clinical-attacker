import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from tools.accuracy_init import init_accuracy_function
from model.loss import multi_label_cross_entropy_loss


class GRNNHA(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(GRNNHA, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.data_size = config.getint("data", "vec_size")
        self.glove = pickle.load(open(config.get("data", "w2v_path"), "rb"))
        self.embedding = nn.Embedding.from_pretrained(
                                    embeddings=torch.from_numpy(self.glove),
                                    freeze = True
                                    )
        self.LSTM = nn.LSTM(input_size = 300,
                            hidden_size = 512,
                            num_layers = 1,
                            bias = True,
                            batch_first = True,
                            bidirectional = False)

        self.fc = nn.Linear(512, self.output_dim)
        self.attn_fc_1 = nn.Linear(512, 256)
        self.attn_fc_2 = nn.Linear(256, 1)
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
        else:
            self.embeded = x.requires_grad_(True)

        #attn part
        all_the_way, (y, _) = self.LSTM(self.embeded)
        attn_weight = self.relu(self.attn_fc_1(all_the_way))
        attn_weight = self.attn_fc_2(attn_weight).permute(0, 2, 1)
        #if softmax
        attn_weight = attn_weight.reshape(attn_weight.shape[0], 2048)
        attn_weight = torch.nn.Softmax(dim=1)(attn_weight)
        attn_weight = attn_weight.reshape(attn_weight.shape[0], 1, 2048)

        y = torch.bmm(attn_weight, all_the_way).reshape(attn_weight.shape[0], 512)

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
            return {"loss": loss, "acc_result": acc_result, "prediction": prediction}

        return {"logits": logits, "prediction": prediction}
