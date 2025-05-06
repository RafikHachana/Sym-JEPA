import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import pdb
from vocab import RemiVocab

class GenreClassificationModel(pl.LightningModule):
    def __init__(self, 
                    num_genres,
                    lr=1e-4,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
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
        self.context_encoder = BertModel(encoder_config)

        self.genre_classifier = nn.Linear(d_model, num_genres)

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


        self.vocab = RemiVocab()

        self.remi_in = nn.Embedding(len(self.vocab), self.d_model)

    def forward(self, input_ids, attention_mask):
        input_ids = self.remi_in(input_ids)
        out = self.context_encoder(inputs_embeds=input_ids, output_hidden_states=True)
        encoder_hidden = out.hidden_states[-1]
        # Mean pool over the sequence length
        encoder_hidden = torch.mean(encoder_hidden, dim=1)
        logits = self.genre_classifier(encoder_hidden)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # pdb.set_trace()
        logits = self(input_ids, None)
        loss = F.cross_entropy(logits, batch['genre_id'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        logits = self(input_ids, None)
        loss = F.cross_entropy(logits, batch['genre_id'])

        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['genre_id']).float().mean()
        self.log('val_acc', acc)
        return loss

    def load_encoder(self, ckpt_path, embedding_path):
        self.context_encoder.load_state_dict(torch.load(ckpt_path))

        self.remi_in.load_state_dict(torch.load(embedding_path))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    