import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=1)
        self.lr = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.lr(x)


class Net(nn.Module):
    """Neural netword with a dueling architecture"""
    def __init__(self, conv_c, strides, h_dims, n_actions):
        super(Net, self).__init__()

        self.convs = nn.ModuleList([
            ConvBlock(in_channels, out_channels, (3, 3), stride)
            for in_channels, out_channels, stride in zip(
                conv_c[:-1], conv_c[1:], strides)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(h_dims[:-1], h_dims[1:])
        ])

        # block in charge of state value estimation
        self.state_block = nn.Sequential(nn.Linear(h_dims[-1], 1))

        # block in charge of advantage value estimation
        self.adv_block = nn.Sequential(nn.Linear(h_dims[-1], n_actions))

    def forward(self, x):
        x.squeeze_(0)
        # Going from (B, 250, 160, 3) -> (B, 3, 250, 160)
        x = x.transpose(-1, -3).transpose(-2, -1)

        for conv in self.convs:
            x = conv(x)

        #print("before flatten ", x.shape)
        x = x.flatten(start_dim=-2)
        #print("after flatten " , x.shape)

        for lin in self.linears:
            x = lin(x)

        v = self.state_block(x)
        a = self.adv_block(x)

        # compute q values:

        q = v + a - a.mean()
        return q
