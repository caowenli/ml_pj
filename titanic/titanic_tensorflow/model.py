import torch


# 定义BP神经网络
class Model(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.out(x))
        return x
