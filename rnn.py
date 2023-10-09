import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import math
from matplotlib import pyplot as plt

num_time_steps = 1000
step_size = 0.1

input_size = 1  # 输入变量的维度，一维
batch_size = 1
hidden_size = 16
num_layers = 1
output_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# use rnn to predict sin
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        for p in self.rnn.parameters():  # 对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


learning_rate = 0.01
model = RNN(input_size, hidden_size, num_layers, output_size)
model.to(device=device)

criterion = nn.MSELoss()
hidden_prev = torch.zeros(num_layers, batch_size, hidden_size, device=device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
x = []
for epoch in range(2000):
    x = [[math.sin(i * step_size)] for i in range(num_time_steps)]
    x = torch.Tensor([x]).to(device=device)
    y = [[math.sin((i + 1) * step_size)] for i in range(num_time_steps)]
    y = torch.Tensor([y]).to(device=device)
    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()
    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: {}, loss: {}".format(epoch, loss.item()))

# test
model.eval()
x = [[math.sin((i) * step_size + 1)] for i in range(100)]
out = x.copy()
hidden_prev = torch.zeros(num_layers, batch_size, hidden_size, device=device)
for epoch in range(num_time_steps):
    output, hidden_prev = model(torch.Tensor([x]).to(device=device), hidden_prev)
    x.pop(0)
    x.append([output.detach()[0][-1][0]])
    out.append([output.detach().to(device='cpu')[0][-1][0]])

# plot the result, the blue line is the prediction, the red line is the real sin value
plt.figure()
plt.plot(np.arange(0, 100, step_size), np.sin(np.arange(0, 100, step_size)), color='r')
plt.plot(np.arange(1, 101, step_size), out[:num_time_steps], color='b')

plt.show()
