import torch
import torch.nn as nn
import torch.functional as F
import json


class AutoEncoder(nn.Module):
    def __init__(self, d_hidden, d_mlp, l1_coeff):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_mlp = d_mlp
        self.l1_coeff = l1_coeff

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(torch.device("cuda"))

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR / (str(version) + ".pt"))
        with open(SAVE_DIR / (str(version) + "_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version):
        cfg = json.load(open(SAVE_DIR / (str(version) + "_cfg.json"), "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR / (str(version) + ".pt")))
        return self
