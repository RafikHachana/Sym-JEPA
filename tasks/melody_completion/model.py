from typing import List, Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import pdb
from vocab import RemiVocab
from model import SymJEPA, SymJEPAPooler
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
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
        self.jepa_pooler = SymJEPAPooler(d_model=d_model)

        self.melody_completion_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
            
        )

        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.xavier_uniform_(self.positional_encoding)



        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation='relu',
        # )
        # # Transformer encoder with one layer
        # self.melody_completion_transformer = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=2
        # )

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.validation_outputs = []

    def forward(self, input_ids, log_similarity=False):
        # with torch.no_grad():
            # first_encoded = self.jepa.encode_context(first)
            # second_encoded = self.jepa.encode_context(second)
        with torch.no_grad():
            embedded = self.jepa.encode_context(input_ids)
        
        pooled = self.jepa_pooler(embedded)

            # first_encoded_mean = first_encoded.mean(dim=1)
            # second_encoded_mean = second_encoded.mean(dim=1)

        # embedded = embedded + self.positional_encoding[:, :embedded.size(1), :]
        # logits = self.melody_completion_classifier(torch.cat((first_encoded_mean, second_encoded_mean), dim=1))
        # if log_similarity:
        #     similarity = torch.nn.functional.cosine_similarity(first_encoded_mean, second_encoded_mean, dim=1)
        #     self.log('first_second_similarity', similarity.mean())


        # hidden_states = self.melody_completion_transformer(embedded)
        logits = self.melody_completion_classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input'], batch['match'])
        loss = F.cross_entropy(logits, batch['match'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input'], batch['match'])
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

        # for uuid in grouped_preds:
        #     grouped_preds[uuid] = torch.stack(grouped_preds[uuid], dim=0)
        #     grouped_preds[uuid] = torch.mean(grouped_preds[uuid], dim=0)
            

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)

        # print("Count of 0s and 1s in the match: ", batch['melody_completion_match'].sum(), batch['melody_completion_match'].size(0) - batch['melody_completion_match'].sum())
        # print("Count of 0s and 1s in the preds: ", preds.sum(), preds.size(0) - preds.sum())
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

        # for c in range(num_classes):
        #     print(f"Class {c} precision: {precision[c]}, recall: {recall[c]}, f1: {f1[c]}")
        
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

        # print("All MAP: ", all_map)
        # print("All Hits at 1: ", all_hits_at_1)
        # print("All Hits at 5: ", all_hits_at_5)
        # print("All Hits at 10: ", all_hits_at_10)
        # print("All Hits at 25: ", all_hits_at_25)

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
    
    