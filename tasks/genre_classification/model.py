import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import pdb
from vocab import RemiVocab
from model import SymJEPA

class GenreClassificationModel(pl.LightningModule):
    def __init__(self, 
                    num_genres,
                    lr=1e-4,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    tokenization='remi',
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

        self.genre_classifier = nn.Linear(d_model, num_genres)

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


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
        loss = F.cross_entropy(logits, batch['genre_id'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        logits = self(input_ids)
        loss = F.cross_entropy(logits, batch['genre_id'])

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['genre_id']).float().mean()
        self.log('val_acc', acc)
        
        # Calculate F1 score manually
        num_classes = logits.size(1)
        tp = torch.zeros(num_classes, device=preds.device)
        fp = torch.zeros(num_classes, device=preds.device)
        fn = torch.zeros(num_classes, device=preds.device)
        
        for c in range(num_classes):
            tp[c] = ((preds == c) & (batch['genre_id'] == c)).sum()
            fp[c] = ((preds == c) & (batch['genre_id'] != c)).sum()
            fn[c] = ((preds != c) & (batch['genre_id'] == c)).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        macro_f1 = f1.mean()
        
        self.log('val_f1', macro_f1)
        return loss

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    