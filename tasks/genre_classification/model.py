import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import pdb
from src.vocab import RemiVocab
from src.model import SymJEPA, SymJEPAPooler
import wandb
from sklearn.metrics import f1_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() * 1e10

class GenreClassificationModel(pl.LightningModule):
    def __init__(self, 
                    num_classes,
                    max_len=2048,
                    lr=1e-3,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    tokenization='remi',
                    class_weights=None,
                    task='genre',
                    use_focal_loss=False,
                 **kwargs):
        super().__init__()

        encoder_config = BertConfig(
        vocab_size=1,
        pad_token_id=0,
        hidden_size=d_model,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
        # position_embedding_type='relative_key_query'
        )

        self.max_len = max_len
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization=tokenization, pass_target_mask_to_predictor=True)
        
        self.jepa.eval()

        self.jepa_pooler = SymJEPAPooler(d_model=d_model)

        self.using_pretrained_encoder = False


        # Add sinusoidal positional encoding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        # nn.init.xavier_uniform_(self.positional_encoding)



        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation='relu',
        # )
        # # Transformer encoder with one layer
        # self.genre_transformer = nn.TransformerEncoder(
        #     encoder_layer,
        #     num_layers=1
        # )

        # self.genre_transformer = BertModel(config=encoder_config)

        self.genre_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, num_classes)
        )

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.class_weights = class_weights.to(self.device) if class_weights is not None else torch.ones(num_classes, device=self.device)
        # self.loss = FocalLoss(weight=self.class_weights) if use_focal_loss else nn.CrossEntropyLoss(weight=self.class_weights)


        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.task = task
        self.class_key = 'genre_id' if task == 'genre' else 'style_id'

    def forward(self, input_ids):
        self.jepa.eval()
        encoder_hidden = self.jepa.encode_context(input_ids)
        genre_hidden = self.jepa_pooler(encoder_hidden)
        logits = self.genre_classifier(genre_hidden)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        logits = self(input_ids)
        loss = self.loss(logits, batch[self.class_key])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        logits = self(input_ids)
        loss = self.loss(logits, batch[self.class_key])

        # print(logits.shape)
        # print(batch[self.class_key].shape)

        self.log('val_loss', loss)
        # print(logits.shape)
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).long()
        acc = (preds == batch[self.class_key]).float().mean()
        # print(acc.shape)
        self.log('val_acc', acc)
        
        # Calculate per-class metrics
        f1 = f1_score(batch[self.class_key].cpu(), preds.cpu(), average='micro')
        self.log('val_f1', f1)

        samples_f1 = f1_score(batch[self.class_key].cpu(), preds.cpu(), average='samples')
        self.log('val_samples_f1', samples_f1)
        
        return loss

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.using_pretrained_encoder = True
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer], [{
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True),
            'monitor': 'train_loss',  # Metric to monitor
            'interval': 'step',
            'frequency': 1
        }]
    
    
