import torch
from torch import nn
from transformers import Trainer  

class RMSETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # RMSE Loss 계산
        loss_fct = nn.MSELoss()
        mse_loss = loss_fct(logits.view(-1), labels.view(-1))
        rmse_loss = torch.sqrt(mse_loss)
        
        return (rmse_loss, outputs) if return_outputs else rmse_loss