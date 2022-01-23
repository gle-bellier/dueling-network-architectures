import torch
import torch.nn as nn


class Net(nn.Module):
    """Neural netword with a dueling architecture"""
    def __init__(self, in_size, h_size, n_actions):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_size, 2 * h_size),
            nn.ReLU(),
            #nn.Linear(h_size, 2*h_size), nn.ReLU()
        )

        # block in charge of state value estimation
        self.state_block = nn.Sequential(nn.Linear(2 * h_size, h_size),
                                         nn.ReLU(), nn.Linear(h_size, 1))

        # block in charge of advantage value estimation
        self.adv_block = nn.Sequential(nn.Linear(2 * h_size, h_size),
                                       nn.ReLU(), nn.Linear(h_size, n_actions))

    def forward(self, x):
        x = self.main(x)

        v = self.state_block(x)
        a = self.adv_block(x)

        # compute q values:

        q = v + a - a.mean()
        return q