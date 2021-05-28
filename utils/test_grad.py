import torch
import torch.nn as nn
import numpy as np



class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2, 1)
        weight = torch.FloatTensor([[1, 2.3], [4, 5.1]])
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

    def forward(self, input):
        x = self.embedding(input)
        return (5 - self.fc(x)) ** 2

    def show(self):
        print(self.fc.weight, self.fc.bias, self.embedding.weight)

input = torch.from_numpy(np.array([1])).cuda().long()
model = LinearRegression().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=1,
                              weight_decay=0)
for idx in range(10):
    loss = model(input)
    print("Current weight:")
    model.show()
    print("Current Loss:")
    print(loss.detach().cpu().numpy())
    print("Curent gradient")
    g = torch.autograd.grad(outputs=loss,
                             inputs=list(model.parameters()),
                             grad_outputs=None,
                             retain_graph=True,
                             create_graph=False,
                             only_inputs=True)
    print(g[0])
    loss.backward()
    optimizer.step()
    print("Modified weight")
    model.show()
    print("-------------------------------------------")
    break
