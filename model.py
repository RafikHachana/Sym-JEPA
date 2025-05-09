import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import math
from dataset import MidiDataModule
from vocab import RemiVocab
from constants import PAD_TOKEN
from transformers import BertConfig, BertModel
from copy import deepcopy
import torch.nn.functional as F

class SymJEPA(pl.LightningModule):
  def __init__(self,
               d_model=512,
               context_size=512,
               tokenization='remi',
               lr=1e-4,
               lr_schedule='linear',
               warmup_steps=100,
               max_steps=None,
               encoder_layers=8,
               predictor_layers=8,
               intermediate_size=2048,
               num_attention_heads=8,
               description_options=None,
               ema=(0.996, 0.999),
               num_epochs=100,
               momentum_start=0.996,
               momentum_end=1.0,
               use_vicreg=False,
               vicreg_sim_weight=25.0,
               vicreg_var_weight=25.0,
               vicreg_cov_weight=1.0,
               vicreg_loss_ratio=0.3,
               pass_target_mask_to_predictor=False,
               **kwargs):
    super().__init__()

    self.tokenization = tokenization
    self.vocab = RemiVocab()
    self.d_model = d_model
    
    # Initialize embeddings based on tokenization method
    if tokenization == 'remi':
        self.remi_in = nn.Embedding(len(self.vocab), self.d_model)
    elif tokenization == 'octuple':
        self.octuple_in = nn.Embedding(len(self.vocab), self.d_model)
        self.octuple_downsampling = nn.Linear(d_model * 8, d_model)
        self.octuple_layer_norm = nn.LayerNorm(d_model)
        self.octuple_dropout = nn.Dropout(0.1)
    else:
        raise ValueError(f"Unknown tokenization method: {tokenization}")

    self.description_options = description_options

    self.context_size = context_size

    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    
    self.max_bars = self.context_size
    self.max_positions = 512

    self.loss_fn = nn.SmoothL1Loss()

    # Store just the EMA parameters
    self.ema = ema

    self.save_hyperparameters()

    self.num_epochs = num_epochs
    self.momentum_start = momentum_start
    self.momentum_end = momentum_end
    self.use_vicreg = use_vicreg
    self.vicreg_sim_weight = vicreg_sim_weight
    self.vicreg_var_weight = vicreg_var_weight
    self.vicreg_cov_weight = vicreg_cov_weight
    self.vicreg_loss_ratio = vicreg_loss_ratio
    self.pass_target_mask_to_predictor = pass_target_mask_to_predictor


    if self.pass_target_mask_to_predictor:
      self.target_mask_embedding = nn.Embedding(2, self.d_model)
    # Enable gradient checkpointing for memory efficiency
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

    predictor_config = deepcopy(encoder_config)
    predictor_config.num_hidden_layers = predictor_layers
    
    # Initialize only the encoder
    self.context_encoder = BertModel(encoder_config)

    self.target_encoder = deepcopy(self.context_encoder)

    self.predictor = BertModel(predictor_config)

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

  def forward(self, context_ids, target_ids=None, return_context_encoder_hidden=False, context_mask=None, target_mask=None):
    # Context and target IDs should be already masked
    if self.tokenization == 'remi':
        context_emb = self.remi_in(context_ids)
    else:  # octuple
        # Embed tokens
        x = self.octuple_in(context_ids)  # [batch_size, seq_len, d_model]
        
        # Reshape to group every 8 tokens
        batch_size, seq_len, emb_dim = x.shape
        x = x.view(batch_size, seq_len // 8, 8 * emb_dim)
        
        # Project grouped embeddings
        context_emb = self.octuple_downsampling(x)
        context_emb = self.octuple_layer_norm(context_emb)
        context_emb = self.octuple_dropout(context_emb)
    out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    encoder_hidden = out.hidden_states[-1]

    if self.tokenization == 'octuple':
      context_mask = context_mask[:, ::8]
    if context_mask is not None:
      encoder_hidden[context_mask] = 0

    if target_ids is not None:
        
        
        with torch.no_grad():
            if self.tokenization == 'remi':
                target_emb = self.remi_in(target_ids)
            else:  # octuple
                x = self.octuple_in(target_ids)
                batch_size, seq_len, emb_dim = x.shape
                x = x.view(batch_size, seq_len // 8, 8 * emb_dim)
                target_emb = self.octuple_downsampling(x)
                target_emb = self.octuple_layer_norm(target_emb)
                target_emb = self.octuple_dropout(target_emb)
                
            out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
            target_encoder_hidden = out.last_hidden_state

        if self.tokenization == 'octuple':
            target_mask = target_mask[:, ::8]
        target_encoder_hidden[target_mask] = 0


        if self.pass_target_mask_to_predictor:
            predictor_input = torch.cat([encoder_hidden, self.target_mask_embedding(target_mask.to(torch.int))], dim=1)
        else:
            predictor_input = encoder_hidden
        pred = self.predictor(inputs_embeds=predictor_input, output_hidden_states=True)
        pred_hidden = pred.last_hidden_state[:, :encoder_hidden.size(1)]

        return pred_hidden, target_encoder_hidden, encoder_hidden
    return pred_hidden
    
  def vicreg_loss(self, context_hidden, target_hidden):
    # Reshape to [batch_size * seq_len, hidden_dim]
    x = context_hidden.view(-1, context_hidden.size(-1))
    y = target_hidden.view(-1, target_hidden.size(-1))
    
    # Invariance loss (similarity)
    sim_loss = F.mse_loss(x, y)
    
    # Variance loss
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    
    # Covariance loss
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    
    # Zero out diagonal elements
    diag_mask = ~torch.eye(cov_x.shape[0], dtype=torch.bool, device=cov_x.device)
    cov_loss = (cov_x[diag_mask]**2).mean() + (cov_y[diag_mask]**2).mean()
    
    # Combine losses with weights
    total_loss = (
        self.vicreg_sim_weight * sim_loss + 
        self.vicreg_var_weight * var_loss + 
        self.vicreg_cov_weight * cov_loss
    )
    
    return total_loss, sim_loss, var_loss, cov_loss

  def get_loss(self, batch):
    pred, target, context_hidden = self(
      batch['context_ids'],
      batch['target_ids'],
      return_context_encoder_hidden=True,
      context_mask=batch.get('context_mask'),
      target_mask=batch.get('target_mask'))
    
    # Original JEPA loss
    jepa_loss = self.loss_fn(pred, target)
    
    if self.use_vicreg:
      vicreg_total, vic_sim, vic_var, vic_cov = self.vicreg_loss(context_hidden, target)

      # Dynamic loss weighting
      with torch.no_grad():
          loss_ratio = vicreg_total / jepa_loss
          scale = self.vicreg_loss_ratio / loss_ratio if loss_ratio > self.vicreg_loss_ratio else 1.0
          
      # Apply dynamic scaling
      vicreg_total = vicreg_total * scale
      
      # Log VicReg components
      self.log('train_vicreg_sim', vic_sim, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      self.log('train_vicreg_var', vic_var, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      self.log('train_vicreg_cov', vic_cov, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      
      # Combine losses
      total_loss = jepa_loss + vicreg_total
      self.log('train_jepa_loss', jepa_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      self.log('train_vicreg_loss', vicreg_total, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      
      return total_loss
    
    return jepa_loss
  
  def training_step(self, batch, batch_idx):
    loss = self.get_loss(batch)

    # Log gradients for both encoders
    target_grad_sum = 0
    context_grad_sum = 0
    
    # Sum target encoder gradients (should be zero)
    for param in self.target_encoder.parameters():
        if param.grad is not None:
            target_grad_sum += param.grad.abs().sum()
            
    # Sum context encoder gradients (should be non-zero)
    for param in self.context_encoder.parameters():
        if param.grad is not None:
            context_grad_sum += param.grad.abs().sum()
            
    # Log both gradient sums
    self.log('target_encoder_grad_sum', target_grad_sum, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    self.log('context_encoder_grad_sum', context_grad_sum, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    # Log current learning rate
    current_lr = self.optimizers().param_groups[0]['lr']
    self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    # Update the target parameters
    with torch.no_grad():
        # Get total batches per epoch from trainer
        total_batches = len(self.trainer.train_dataloader)
        total_steps = total_batches * self.num_epochs
        current_step = self.current_epoch * total_batches + batch_idx
        
        # Linear interpolation between ema[0] and ema[1]
        m = self.ema[0] + (self.ema[1] - self.ema[0]) * (current_step / total_steps)
        m = min(m, self.ema[1])  # Ensure we don't exceed the maximum value
        
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        self.log('target_momentum', m, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
    self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    return loss
  
  def validation_step(self, batch, batch_idx):
    loss = self.get_loss(batch)
    
    # Get encoder outputs
    context_emb = self.remi_in(batch['context_ids'])
    context_out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    context_hidden = context_out.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
    
    with torch.no_grad():
        target_emb = self.remi_in(batch['target_ids'])
        target_out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
        target_hidden = target_out.hidden_states[-1]
        
        # Calculate representation collapse metrics
        
        # 1. Cosine similarity (higher value indicates more collapse)
        cos_sim = F.cosine_similarity(
            context_hidden.view(-1, context_hidden.size(-1)),
            target_hidden.view(-1, target_hidden.size(-1)),
            dim=1
        ).mean()
        
        # 2. Mean squared difference of normalized representations (lower value indicates more collapse)
        context_norm = F.normalize(context_hidden, p=2, dim=-1)
        target_norm = F.normalize(target_hidden, p=2, dim=-1)
        mse_diff = torch.mean((context_norm - target_norm) ** 2)
        
        # 3. Variance of representations across batch
        # Reshape to [batch_size * seq_len, hidden_dim]
        context_flat = context_hidden.view(-1, context_hidden.size(-1))
        target_flat = target_hidden.view(-1, target_hidden.size(-1))
        
        # Calculate variance for each feature across batch
        context_var = torch.var(context_flat, dim=0).mean()  # Mean variance across features
        target_var = torch.var(target_flat, dim=0).mean()
        
        # Log all metrics
        self.log('val_cosine_similarity', cos_sim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mse_diff', mse_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_context_variance', context_var, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_target_variance', target_var, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Also log the original norms
        context_norm = torch.norm(context_hidden, dim=-1).mean()
        target_norm = torch.norm(target_hidden, dim=-1).mean()
        self.log('val_context_encoder_norm', context_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_target_encoder_norm', target_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
