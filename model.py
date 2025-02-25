import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import math
from dataset import MidiDataModule
from vocab import RemiVocab
from constants import PAD_TOKEN
from transformers import BertConfig, BertModel
from copy import deepcopy

class SymJEPA(pl.LightningModule):
  def __init__(self,
               d_model=256,
               d_latent=256,
               context_size=512,
               lr=1e-4,
               lr_schedule='sqrt_decay',
               warmup_steps=100,
               max_steps=None,
               encoder_layers=6,
               intermediate_size=2048,
               num_attention_heads=8,
               description_options=None,
               ema=(0.996, 0.999),
               ipe=1000,
               ipe_scale=1,
               num_epochs=100):
    super(SymJEPA, self).__init__()

    self.description_options = description_options



    self.context_size = context_size
    self.d_model = d_model
    self.d_latent = d_latent

    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    
    self.vocab = RemiVocab()

    encoder_config = BertConfig(
      vocab_size=1,
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=encoder_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )
    

    # Initialize only the encoder
    self.context_encoder = BertModel(encoder_config)

    self.target_encoder = deepcopy(self.context_encoder)

    self.predictor = BertModel(encoder_config)


    self.max_bars = self.context_size
    self.max_positions = 512

    self.remi_in = nn.Embedding(len(self.vocab), self.d_model)
    
    self.loss_fn = nn.SmoothL1Loss()

    self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))
        
    self.save_hyperparameters()

  def get_datamodule(self, midi_files, **kwargs):
    return MidiDataModule(
      midi_files, 
      self.context_size,
      description_flavor=self.description_flavor,
      max_bars=self.max_bars,
      max_positions=self.max_positions,
      description_options=self.description_options,
      **kwargs
    )

  def forward(self, z, encode_target=False):
    z_emb = self.remi_in(z)
    out = self.context_encoder(inputs_embeds=z_emb, output_hidden_states=True)
    encoder_hidden = out.hidden_states[-1]
    pred = self.predictor(inputs_embeds=z_emb, output_hidden_states=True)
    pred = pred.last_hidden_state
    if encode_target:
      with torch.no_grad():
        out = self.target_encoder(inputs_embeds=z_emb, output_hidden_states=True)
        target_encoder_hidden = out.last_hidden_state
      return pred, target_encoder_hidden
    return encoder_hidden
    
  def get_loss(self, batch):
    x = batch['input_ids']
    pred, target = self(x, encode_target=True)
    
    # Compute loss with the tensor outputs
    loss = self.loss_fn(pred, target)
    
    return loss
  
  def training_step(self, batch, batch_idx):
    loss = self.get_loss(batch)

    # Update the target parameters
    with torch.no_grad():
      m = next(self.momentum_scheduler)
      for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
          param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

      self.log('target_momentum', m, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
    self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    return loss
  
  def validation_step(self, batch, batch_idx):
    loss = self.get_loss(batch)
    self.log('valid_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    return self.get_loss(batch)
        
  def configure_optimizers(self):
    # set LR to 1, scale with LambdaLR scheduler
    optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)

    if self.lr_schedule == 'sqrt_decay':
      # constant warmup, then 1/sqrt(n) decay starting from the initial LR
      lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1)/self.warmup_steps))
    elif self.lr_schedule == 'linear':
      # linear warmup, linear decay
      lr_func = lambda step: min(self.lr, self.lr*step/self.warmup_steps, self.lr*(1 - (step - self.warmup_steps)/self.max_steps))
    elif self.lr_schedule == 'cosine':
      # linear warmup, cosine decay to 10% of initial LR
      lr_func = lambda step: self.lr * min(step/self.warmup_steps, 0.55 + 0.45*math.cos(math.pi*(min(step, self.max_steps) - self.warmup_steps)/(self.max_steps - self.warmup_steps)))
    else:
      # Use no lr scheduling
      lr_func = lambda step: self.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    return [optimizer], [{
      'scheduler': scheduler,
      'interval': 'step',
    }]
