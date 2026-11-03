import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, target_norm_scale):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.act = nn.GELU()
        self.norm = nn.RMSNorm(out_dim)
        self.target_norm_scale = target_norm_scale

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        if self.norm is not None:
            x = self.norm(x) * self.target_norm_scale
        return x # norm(l2(act(f1(x))))