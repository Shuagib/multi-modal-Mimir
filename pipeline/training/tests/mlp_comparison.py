import torch
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from torch.utils.data import DataLoader
from projector.two_layer_mlp import MLP

c = ProjectorTrainConfig()
eval_data = FeatureDataset(c.eval_dir)
eval_load = DataLoader(eval_data, batch_size=c.batch_size, shuffle=True)

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(c.device)


with torch.no_grad():
    b = next(iter(eval_load))
    f = b["features"].to(c.device).float()
    out = mlp(f).flatten(1)
    cos_in  = torch.nn.functional.cosine_similarity(f.flatten(1)[0], f.flatten(1)[1], dim=0)
    cos_out = torch.nn.functional.cosine_similarity(out[0], out[1], dim=0)
    print(f"SigLIP cos {cos_in:.3f}  MLP cos {cos_out:.3f}")