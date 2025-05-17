import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import pdb
from vocab import RemiVocab
from model import SymJEPA
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix

class MelodyCompletionModel(pl.LightningModule):
    def __init__(self, 
                    max_len=512,
                    lr=1e-3,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    tokenization='remi',
                    class_weights=None,
                 **kwargs):
        super().__init__()

        # encoder_config = BertConfig(
        # vocab_size=1,
        # pad_token_id=0,
        # hidden_size=d_model,
        # num_hidden_layers=encoder_layers,
        # num_attention_heads=num_attention_heads,
        # intermediate_size=intermediate_size,
        # max_position_embeddings=1024,
        # position_embedding_type='relative_key_query'
        # )

        self.max_len = max_len
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization=tokenization, pass_target_mask_to_predictor=True)

        self.melody_completion_classifier = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
            
        )

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def forward(self, first, second):
        with torch.no_grad():
            first_encoded = self.jepa.encode_context(first)
            second_encoded = self.jepa.encode_context(second)

            first_encoded_mean = first_encoded.mean(dim=1)
            second_encoded_mean = second_encoded.mean(dim=1)

        logits = self.melody_completion_classifier(torch.cat((first_encoded_mean, second_encoded_mean), dim=1))
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['melody_completion_start'], batch['melody_completion_end'])
        loss = F.cross_entropy(logits, batch['melody_completion_match'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['melody_completion_start'], batch['melody_completion_end'])
        loss = F.cross_entropy(logits, batch['melody_completion_match'])

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['melody_completion_match']).float().mean()
        self.log('val_acc', acc)
        
        # Calculate per-class metrics
        num_classes = logits.size(1)
        tp = torch.zeros(num_classes, device=preds.device)
        fp = torch.zeros(num_classes, device=preds.device)
        fn = torch.zeros(num_classes, device=preds.device)
        
        for c in range(num_classes):
            tp[c] = ((preds == c) & (batch['melody_completion_match'] == c)).sum()
            fp[c] = ((preds == c) & (batch['melody_completion_match'] != c)).sum()
            fn[c] = ((preds != c) & (batch['melody_completion_match'] == c)).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Log per-class metrics
        # for c in range(num_classes):
        #     self.log(f'val_precision_class_{c}', precision[c])
        #     self.log(f'val_recall_class_{c}', recall[c])
        #     self.log(f'val_f1_class_{c}', f1[c])
        
        # Log macro averages
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        self.log('val_precision', macro_precision)
        self.log('val_recall', macro_recall)
        self.log('val_f1', macro_f1)
        
        
        # Multi-class case
        # roc_auc = torch.tensor(roc_auc_score(batch['genre_id'].cpu(), F.softmax(logits, dim=1).cpu(), multi_class='ovr'))
        # self.log('val_roc_auc', roc_auc)
        
        # Log confusion matrix to wandb
        # cm = confusion_matrix(batch['melody_completion_match'].cpu(), preds.cpu())
        # self.logger.experiment.log({
        #     "confusion_matrix": wandb.plot.confusion_matrix(
        #         probs=None,
        #         y_true=batch['melody_completion_match'].cpu().numpy(),
        #         preds=preds.cpu().numpy(),
        #         class_names=[f"Class_{i}" for i in range(num_classes)]
        #     )
        # })
        
        return loss

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    