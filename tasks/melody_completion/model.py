import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SymJEPA, SymJEPAPooler
from sklearn.metrics import average_precision_score
import numpy as np


class MelodyCompletionModel(pl.LightningModule):
    def __init__(self, 
                    max_len=2048,
                    lr=1e-3,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    tokenization='remi',
                    class_weights=None,
                 **kwargs):
        super().__init__()

        self.max_len = max_len
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization=tokenization, pass_target_mask_to_predictor=True)
        self.jepa.eval()
        self.jepa_pooler = SymJEPAPooler(d_model=d_model)

        self.melody_completion_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 2)
        )

        prior = 0.4
        bias_val = math.log(prior/(1-prior))
        self.melody_completion_classifier[2].bias.data.fill_(bias_val)

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.validation_outputs = []

    def forward(self, input_ids, log_similarity=False):
        self.jepa.eval()
        with torch.no_grad():
            embedded = self.jepa.encode_context(input_ids)
        
        pooled = self.jepa_pooler(embedded)

        logits = self.melody_completion_classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input'])
        loss = F.cross_entropy(logits, batch['match'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input'])
        loss = F.cross_entropy(logits, batch['match'])

        probs = F.softmax(logits, dim=1)

        grouped_preds = {}
        grouped_targets = {}
        for i, uuid in enumerate(batch['uuid']):
            if uuid not in grouped_preds:
                grouped_preds[uuid] = []
                grouped_targets[uuid] = []
            # Get the probability of the positive class
            grouped_preds[uuid].append(probs[i][1].item())
            grouped_targets[uuid].append(batch['match'][i].item())
            

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == batch['match']).float().mean()
        self.log('val_acc', acc)
        
        # Calculate per-class metrics
        num_classes = logits.size(1)
        tp = torch.zeros(num_classes, device=preds.device)
        fp = torch.zeros(num_classes, device=preds.device)
        fn = torch.zeros(num_classes, device=preds.device)
        
        for c in range(num_classes):
            tp[c] = ((preds == c) & (batch['match'] == c)).sum()
            fp[c] = ((preds == c) & (batch['match'] != c)).sum()
            fn[c] = ((preds != c) & (batch['match'] == c)).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        
        # Log macro averages
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        self.log('val_precision', macro_precision)
        self.log('val_recall', macro_recall)
        self.log('val_f1', macro_f1)
        
        
        self.validation_outputs.append({
            'loss': loss,
            'grouped_preds': grouped_preds,
            'grouped_targets': grouped_targets
        })

        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        all_map =[]

        all_hits_at_1 = []
        all_hits_at_5 = []
        all_hits_at_10 = []
        all_hits_at_25 = []

        for output in outputs:
            for uuid in output['grouped_preds']:
                combined = list(zip(output['grouped_preds'][uuid], output['grouped_targets'][uuid]))
                combined.sort(key=lambda x: x[0], reverse=True)

                combined = np.array(combined)
                
                # Calculate Mean Average Precision
                ap = self.calculate_ap(combined)
                # print("AP: ", ap)
                all_map.append(ap)

                all_hits_at_1.append(self.hits_at_k(combined, 1))
                all_hits_at_5.append(self.hits_at_k(combined, 5))
                all_hits_at_10.append(self.hits_at_k(combined, 10))
                all_hits_at_25.append(self.hits_at_k(combined, 25))

        self.log('val_map', torch.tensor(all_map).mean())
        self.log('val_hits_at_1', torch.tensor(all_hits_at_1).mean())
        self.log('val_hits_at_5', torch.tensor(all_hits_at_5).mean())
        self.log('val_hits_at_10', torch.tensor(all_hits_at_10).mean())
        self.log('val_hits_at_25', torch.tensor(all_hits_at_25).mean())

        self.validation_outputs = []

    def calculate_ap(self, combined):
        return average_precision_score(y_true=combined[:, 1], y_score=combined[:, 0])

    def hits_at_k(self, combined, k):
        """
        combined: np.ndarray, should be sorted by the first column
        k: int
        """
        return np.sum(combined[:k, 1])

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    