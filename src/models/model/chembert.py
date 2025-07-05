import torch
import torch.nn as nn 
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification

class ChemBERT(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", mol_f_dim=2083, out_dim=1) -> None:
        super(ChemBERT, self).__init__()

        self.ChemBERT_encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )

        self.projection = nn.Linear(mol_f_dim, 1024)  # 1 + 2082 = 2083
        self.ln = nn.LayerNorm(1024)  # LayerNorm for the projection output
        self.out = nn.Linear(1024, out_dim)

        self.act = nn.SELU()
        self.drop = nn.Dropout(0.1)
        
    def forward(self, input_ids=None, attention_mask=None, mol_f=None, labels=None, **kwargs):
        enc_out = self.ChemBERT_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        if mol_f is not None:
            h = torch.cat([enc_out, mol_f], dim=1)
        else:
            h = enc_out

        h = self.projection(h)
        h = self.ln(h)
        h = self.act(h)
        logits = self.out(h).squeeze(-1)

        return {
            "logits": logits
        }
