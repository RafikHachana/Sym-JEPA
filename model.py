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
               num_epochs=1):
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

  def forward(self, context_ids, target_ids=None):
    context_emb = self.remi_in(context_ids)
    out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    encoder_hidden = out.hidden_states[-1]

    if target_ids is not None:
      pred = self.predictor(inputs_embeds=encoder_hidden, output_hidden_states=True)
      pred_hidden = pred.last_hidden_state
      with torch.no_grad():
        target_emb = self.remi_in(target_ids)
        out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
        target_encoder_hidden = out.last_hidden_state

      return pred_hidden, target_encoder_hidden
    return encoder_hidden
    
  def get_loss(self, batch):
    pred, target = self(batch['context_ids'], batch['target_ids'])
    
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

    # Get encoder outputs for norm calculation
    context_emb = self.remi_in(batch['context_ids'])
    context_out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    context_hidden = context_out.hidden_states[-1]
    
    with torch.no_grad():
        target_emb = self.remi_in(batch['target_ids'])
        target_out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
        target_hidden = target_out.hidden_states[-1]
        
        # Calculate mean norms
        context_norm = torch.norm(context_hidden, dim=-1).mean()
        target_norm = torch.norm(target_hidden, dim=-1).mean()
        
        # Log the norms
        self.log('val_context_encoder_norm', context_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_target_encoder_norm', target_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
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
