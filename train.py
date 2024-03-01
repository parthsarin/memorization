import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class NonLinearRNN(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super().__init__()

        # hidden dynamics
        self.Wx = nn.Parameter(torch.randn(hidden_dim, visible_dim))
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.randn(hidden_dim))

        # output dynamics
        self.Wy = nn.Parameter(torch.randn(visible_dim, hidden_dim))
        self.by = nn.Parameter(torch.randn(visible_dim))

    def forward(self, x, h):
        h = torch.matmul(self.Wx, x) + torch.matmul(self.Wh, h) + self.bh
        y = torch.matmul(self.Wy, h) + self.by
        y = F.softmax(y, dim=0)
        return y, h


# training set generation
def encode(s, n_chars=None):
    """
    Character k is represented as the kth row of the identity matrix.
    """
    if n_chars is None:
        n_chars = len(set(s))
    e_dict = {ch: i for i, ch in enumerate(sorted(set(s)))}
    S = torch.eye(n_chars)
    return torch.stack([S[:, e_dict[ch]] for ch in s]).T


hidden_dim = 2
data = [
    encode("aabbaaaaabbb" * 10)
]
model = NonLinearRNN(data[0].shape[0], hidden_dim)


def train():
    # Adam optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1_000_000):
        loss = 0
        for p in data:
            h = torch.Tensor([1] + [0] * (hidden_dim - 1))
            for i in range(p.shape[1] - 1):
                x = p[:, i]
                y = p[:, i + 1]
                y_pred, h = model(x, h)

                # L2 loss
                loss += torch.sum((y_pred - y) ** 2)

                # cross entropy loss
                # loss += -torch.sum(y * torch.log(y_pred))
                if torch.isnan(loss):
                    print("loss is nan")
                    exit(1)

        # print loss every 10 epochs
        if epoch % 100 == 0:
            l = loss.item()
            print(f"[epoch {epoch}] loss: {l}")

            # save model
            torch.save(model.state_dict(), "model.pt")

            if l < 1:
                break

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()


def visualize():
    hidden_states = [torch.Tensor([1] + [0] * (hidden_dim - 1))]

    h = hidden_states[0]
    x = data[0][:, 0]
    for _ in range(100):
        y_pred, h = model(x, h)
        print(("a", "b")[torch.argmax(y_pred).item()], end="")

        # step 2
        x = y_pred
        hidden_states.append(h)

    # print out the model parameters, with their names
    print("\n\nmodel parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param}")

    # create a heatmap of locations based on the model's probability y[0]
    # (i.e. the probability of being in state 0)
    def prob(h):
        return F.softmax(torch.matmul(model.Wy, h) + model.by)[0]

    # loop over a meshgrid
    x = np.linspace(min(h[0] for h in hidden_states).item(), max(h[0]
                    for h in hidden_states).item(), 100)
    y = np.linspace(min(h[1] for h in hidden_states).item(), max(h[1]
                    for h in hidden_states).item(), 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            h = torch.Tensor([xx[i, j], yy[i, j]])
            grid[i, j] = prob(h).item()

    # plot the hidden states
    plt.plot([h[0].item() for h in hidden_states],
             [h[1].item() for h in hidden_states],
             color="black")

    # scatter plot each of the points with a color corresponding to the model's
    # probability of being in state 0
    plt.contourf(x, y, grid, levels=100, cmap="coolwarm")

    # add a legend for color
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    # train()
    model.load_state_dict(torch.load("model.pt"))

    visualize()
