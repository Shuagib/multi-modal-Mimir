from torch import nn


class MLP(nn.Module):
    def __init__(self, out_dim, target_norm_scale):
        super().__init__()
        # self.layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)
        # self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.layer1 = nn.Linear(in_features=768, out_features=1536, bias=False)
        self.layer2 = nn.Linear(in_features=256, out_features=24)

        #self.layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        #self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=False)
        self.act = nn.GELU()
        self.norm = nn.RMSNorm(out_dim)
        self.target_norm_scale = target_norm_scale

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = x.transpose(-1,-2)
        x = self.layer2(x)
        x = x.transpose(-1,-2)
        if self.norm is not None:
            x = self.norm(x) * self.target_norm_scale
        return x # norm(l2(act(f1(x))))