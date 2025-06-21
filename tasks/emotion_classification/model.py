import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SymJEPA, SymJEPAPooler
from sklearn.metrics import average_precision_score
import numpy as np


class EmotionClassifier(pl.LightningModule):
    def __init__(self, 
                max_len=2048,
                lr=1e-3,
                d_model=512,
                tokenization='octuple',
                num_classes=2,
                class_key="emotion_quadrant",
                **kwargs):
        super().__init__()

        self.max_len = max_len
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization=tokenization, pass_target_mask_to_predictor=True)
        self.jepa.eval()
        self.jepa_pooler = SymJEPAPooler(d_model=d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, num_classes)
        )

        self.num_classes = num_classes

        self.lr = lr
        self.class_key = class_key

    def forward(self, input_ids):
        self.jepa.eval()
        embedded = self.jepa.encode_context(input_ids)
        
        pooled = self.jepa_pooler(embedded)

        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch[self.class_key])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch[self.class_key])

            

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == batch[self.class_key]).float().mean()
        self.log('val_acc', acc)
        
        # Calculate per-class metrics
        num_classes = logits.size(1)
        tp = torch.zeros(num_classes, device=preds.device)
        fp = torch.zeros(num_classes, device=preds.device)
        fn = torch.zeros(num_classes, device=preds.device)
        
        for c in range(num_classes):
            tp[c] = ((preds == c) & (batch[self.class_key] == c)).sum()
            fp[c] = ((preds == c) & (batch[self.class_key] != c)).sum()
            fn[c] = ((preds != c) & (batch[self.class_key] == c)).sum()
        
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
        

        return loss

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
