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

class GenreClassificationModel(pl.LightningModule):
    def __init__(self, 
                    num_classes,
                    lr=1e-3,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    tokenization='remi',
                    class_weights=None,
                    task='genre',
                 **kwargs):
        super().__init__()

        encoder_config = BertConfig(
        vocab_size=1,
        pad_token_id=0,
        hidden_size=d_model,
        num_hidden_layers=encoder_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=1024,
        position_embedding_type='relative_key_query'
        )
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization=tokenization, pass_target_mask_to_predictor=True)

        self.genre_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.class_weights = class_weights.to(self.device) if class_weights is not None else torch.ones(num_classes, device=self.device)

        self.task = task
        self.class_key = 'genre_id' if task == 'genre' else 'style_id'

    def forward(self, input_ids):
        encoder_hidden = self.jepa.encode_context(input_ids)
        # Mean pool over the sequence length
        encoder_hidden = torch.mean(encoder_hidden, dim=1)
        logits = self.genre_classifier(encoder_hidden)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # pdb.set_trace()
        logits = self(input_ids)
        loss = F.cross_entropy(logits, batch[self.class_key],
        weight=self.class_weights.to(self.device)
        )
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        logits = self(input_ids)
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
        cm = confusion_matrix(batch[self.class_key].cpu(), preds.cpu())
        self.logger.experiment.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=batch[self.class_key].cpu().numpy(),
                preds=preds.cpu().numpy(),
                class_names=[f"Class_{i}" for i in range(num_classes)]
            )
        })
        
        return loss

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    