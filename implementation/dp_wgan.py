# import table
import torch.nn as nn


# GAN Generator class
class Generator(nn.Module):

    def __init__(self, input_dim, output_dim, binary=True, device='cpu'):
        super(Generator, self).__init__()

        # linear layer block
        def block(inp, out, Activation, device):
            return nn.Sequential(
                nn.Linear(inp, out, bias=False),
                nn.LayerNorm(out),
                Activation(),
            ).to(device)

        # block structure
        self.block_0 = block(input_dim, input_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)
        self.block_1 = block(input_dim, input_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)
        self.block_2 = block(input_dim, output_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2), device)

    # forward pass
    def forward(self, x):
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x = self.block_2(x)
        return x


# GAN Discriminator class
class Discriminator(nn.Module):

    def __init__(self, input_dim, device='cpu'):
        super(Discriminator, self).__init__()

        # network structure
        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        ).to(device)

    # forward pass
    def forward(self, x):
        return self.model(x)
