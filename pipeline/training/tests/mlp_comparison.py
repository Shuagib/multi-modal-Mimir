import torch
from torch.utils.data import DataLoader

from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from projector.two_layer_mlp import MLP

c = ProjectorTrainConfig()
eval_data = FeatureDataset(c.eval_dir)
eval_load = DataLoader(eval_data, batch_size=c.batch_size, shuffle=True)

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(c.device)

@torch.no_grad()
def main():
    torch.manual_seed(0)
    mlp.load_state_dict(torch.load(c.checkpoint_dir / "mlp.pt", map_location=c.device))
    mlp.eval()
    for _ in range(10):
        i, j = torch.randperm(len(eval_data))[:2].tolist()
        print(f"pair ({i}, {j})")
        f = torch.stack([eval_data[i]["features"], eval_data[j]["features"]]).to(c.device).float()
        out = mlp(f)
        nw_in, nw_out = norm_wasserstein(f, out)
        out, f = out.flatten(1), f.flatten(1)
        cos_in, cos_out = cosine_sim(f, out)
        kl_in, kl_out = kl_div(f, out)
        print(f"SigLIP co-similarity {cos_in:.3f}  MLP cos-similiarity {cos_out:.3f}")
        print(f"SigLIP KL-divergence {kl_in:.3f}  MLP KL-divergence {kl_out:.3f}")
        print(f"SigLIP norm-wasserstein {nw_in:.3f}  MLP norm-wasserstein {nw_out:.3f}")

def cosine_sim(features, out):
    cos_in  = torch.nn.functional.cosine_similarity(features[0], features[1], dim=0)
    cos_out = torch.nn.functional.cosine_similarity(out[0], out[1], dim=0)
    return cos_in, cos_out

def kl_div(features, out):
    # scale to conserve ratio
    def std(x):
        return (x - x.mean()) / x.std()

    in1 = torch.softmax(std(features[0]), dim=0)      # image 0 as a distribution over dimensions
    in2 = torch.softmax(std(features[1]), dim=0)      # image 1
    kl_in = (in1 * (in1.log() - in2.log())).sum()

    out1 = torch.softmax(std(out[0]), dim=0)
    out2 = torch.softmax(std(out[1]), dim=0)
    kl_out = (out1 * (out1.log() - out2.log())).sum()

    return kl_in, kl_out

def norm_wasserstein(features, out, n_proj=256):
    nw_in1 = features[0]
    nw_in2 = features[1]
    n = min(len(nw_in1), len(nw_in2))
    nw_in1 = nw_in1[torch.randperm(len(nw_in1), device=nw_in1.device)[:n]]
    nw_in2 = nw_in2[torch.randperm(len(nw_in2), device=nw_in2.device)[:n]]

    dirs_in = torch.nn.functional.normalize(torch.randn(nw_in1.shape[1], n_proj, device=nw_in1.device), dim=0)

    p_in1, _ = torch.sort(nw_in1 @ dirs_in, dim=0)
    p_in2, _ = torch.sort(nw_in2 @ dirs_in, dim=0)
    
    w_in = (p_in1 - p_in2).abs().mean()
    nw_in = w_in / (p_in2.std() + 1e-8)     # nw_in

    nw_out1 = out[0]
    nw_out2 = out[1]
    nw_out1 = nw_out1[torch.randperm(len(nw_out1), device=nw_out1.device)[:n]]
    nw_out2 = nw_out2[torch.randperm(len(nw_out2), device=nw_out2.device)[:n]]

    dirs_out = torch.nn.functional.normalize(torch.randn(nw_out1.shape[1], n_proj, device=nw_out1.device), dim=0)

    p_out1, _ = torch.sort(nw_out1 @ dirs_out, dim=0)
    p_out2, _ = torch.sort(nw_out2 @ dirs_out, dim=0)

    w_out = (p_out1 - p_out2).abs().mean()
    nw_out = w_out / (p_out2.std() + 1e-8)     # nw_out

    return nw_in, nw_out

if __name__ == "__main__":
    main()

    
