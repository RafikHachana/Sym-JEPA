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

from octuple_tokenizer import token_to_value, get_max_vector, OctupleVocab, max_pitch, bar_max, max_inst, ts_list
from positional_encoding import MusicPositionalEncoding, FundamentalMusicEmbedding
import pdb

class Utils:
    @staticmethod
    def decode_tokens(tokens, vocab):
        """
        Returns a normalized set of 8-dim vectors
        """
        # print(tokens.shape)
        unnormalized = torch.tensor([[token_to_value(token) for token in seq] for seq in map(vocab.decode, tokens.cpu())])
        # print("Unnormalized: ", unnormalized.shape)
        sequences = list(map(vocab.decode, tokens.cpu()))
        # print("Percentage of <mask>: ", sum(seq.count("<mask>") for seq in sequences) / sum(len(seq) for seq in sequences))
        # print("Percentage of -1: ", (unnormalized == -1).sum() / unnormalized.numel())
        unnormalized = unnormalized.view(tokens.shape[0], -1, 8)
        # print("Unnormalized: ", unnormalized.shape)
        max_vector = torch.tensor(get_max_vector(), dtype=torch.float)
        return unnormalized / max_vector

    @staticmethod
    def get_bar_sequence(decoded_tokens):
        return decoded_tokens[:, :, 0]

    @staticmethod
    def get_local_onset_sequence(decoded_tokens):
        return decoded_tokens[:, :, 1]

    @staticmethod
    def get_duration_sequence(decoded_tokens):
        return decoded_tokens[:, :, 4]

    @staticmethod
    def get_pitch_sequence(decoded_tokens):
        pitch_sequence = decoded_tokens[:, :, 3].clone()
        pitch_sequence[pitch_sequence > max_pitch] = -1
        return pitch_sequence

    @staticmethod
    def get_continuous_token_size():
        return int(ceil(log2(max_bars))) + 5 + 12 + max_pitch + max_inst + len(ts_list)
    @staticmethod
    def get_continuous_tokens(decoded_tokens):
        bar = decoded_tokens[:, :, 0]
        bar_binary_rep_size = int(ceil(log2(max_bars)))
        bar_binary_representation = torch.zeros(bar.shape[0], bar.shape[1], bar_binary_rep_size, device=bar.device)
        for i in range(bar_binary_rep_size):
            bar_binary_representation[:, :, i] = (bar >> i) & 1

        octave = torch.where(decoded_tokens[:, :, 3] > max_pitch, -1, decoded_tokens[:, :, 3] // 12)
        tempo = decoded_tokens[:, :, 7]
        duration = decoded_tokens[:, :, 4]
        local_onset = decoded_tokens[:, :, 1]
        velocity = decoded_tokens[:, :, 5]

        pitch_class = torch.where(decoded_tokens[:, :, 3] > max_pitch, 0, decoded_tokens[:, :, 3] % 12 + 1)
        pitch_class_ohe = F.one_hot(pitch_class, num_classes=12)

        drum_pitch = torch.where(decoded_tokens[:, :, 3] <= max_pitch, 0, decoded_tokens[:, :, 3] + 1)
        drum_pitch_ohe = F.one_hot(drum_pitch, num_classes=max_pitch)

        instrument = decoded_tokens[:, :, 2]
        instrument_ohe = F.one_hot(instrument, num_classes=max_inst)


        time_signature = decoded_tokens[:, :, 6]
        time_signature_ohe = F.one_hot(time_signature, num_classes=len(ts_list))

        continuous_tokens = torch.cat([bar_binary_representation, octave, tempo, duration, local_onset, velocity, pitch_class_ohe, drum_pitch_ohe, instrument_ohe, time_signature_ohe], dim=-1)

        return continuous_tokens





class SymJEPA(pl.LightningModule):
  def __init__(self,
               d_model=512,
               context_size=2048,
               tokenization='remi',
               lr=1e-4,
               lr_schedule='linear',
               warmup_steps=100,
               max_steps=None,
               encoder_layers=16,
               predictor_layers=4,
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
               fuse_decoded_tokens=True,
               add_onset_positional_encoding=True,
               fuse_fme=True,
               use_custom_continuous_tokens=False,
               **kwargs):
    super().__init__()

    self.tokenization = tokenization
    self.vocab = RemiVocab() if tokenization == 'remi' else OctupleVocab()
    self.d_model = d_model

    
    # Initialize embeddings based on tokenization method
    if tokenization == 'remi':
        self.remi_in = nn.Embedding(len(self.vocab), self.d_model)
        if fuse_decoded_tokens or add_onset_positional_encoding or fuse_fme:
           raise ValueError("Remi tokenization does not support fused tokens, onset positional encoding, or FME")
    elif tokenization == 'octuple':
        self.octuple_in = nn.Embedding(len(self.vocab), self.d_model)
        self.octuple_downsampling = nn.Linear(d_model * 8, d_model)
        self.octuple_layer_norm = nn.LayerNorm(d_model)
        self.octuple_dropout = nn.Dropout(0.1)


        if fuse_decoded_tokens:
            self.decoded_tokens_in = nn.Linear(8, d_model)
            self.decoded_tokens_layer_norm = nn.LayerNorm(d_model)
            self.decoded_tokens_dropout = nn.Dropout(0.1)

            self.fusion_layer = nn.Linear(d_model * 2, d_model)
            self.fusion_layer_norm = nn.LayerNorm(d_model)
            self.fusion_dropout = nn.Dropout(0.1)

        if fuse_fme:
            self.pitch_fme = FundamentalMusicEmbedding(d_model=d_model, base=9919, device='cuda', type="se", if_trainable=True, translation_bias_type="nd", emb_nn=True)
            self.duration_fme = FundamentalMusicEmbedding(d_model=d_model, base=7920, device='cuda', type="se", if_trainable=True, translation_bias_type="nd", emb_nn=True)

            self.fuse_pitch_duration_layer = nn.Linear(d_model * 2, d_model)

            self.fuse_fme_layer = nn.Linear(d_model * 2, d_model)
            self.fuse_fme_layer_norm = nn.LayerNorm(d_model)
            self.fuse_fme_dropout = nn.Dropout(0.1)

        if use_custom_continuous_tokens:
            continuous_token_size = Utils.get_continuous_token_size()
            self.continuous_tokens_in = nn.Linear(continuous_token_size, d_model)
            self.continuous_tokens_layer_norm = nn.LayerNorm(d_model)
            self.continuous_tokens_dropout = nn.Dropout(0.1)

            self.fuse_custom_continuous_tokens_layer = nn.Linear(d_model * 2, d_model)
            self.fuse_custom_continuous_tokens_layer_norm = nn.LayerNorm(d_model)
            self.fuse_custom_continuous_tokens_dropout = nn.Dropout(0.1)

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
    self.fuse_decoded_tokens = fuse_decoded_tokens
    self.add_onset_positional_encoding = add_onset_positional_encoding
    self.fuse_fme = fuse_fme
    self.use_custom_continuous_tokens = use_custom_continuous_tokens
    self.positional_encoding = nn.Parameter(torch.zeros(1, 768, d_model))
    nn.init.xavier_uniform_(self.positional_encoding)

    if self.add_onset_positional_encoding:
        self.onset_positional_encoding = MusicPositionalEncoding(
          d_model=d_model,
          dropout=0.1,
          max_len=1024,
          if_index=True,
          if_global_timing=True,
          if_modulo_timing=True,
          device='cuda')
    else:
        self.onset_positional_encoding = None

    if self.pass_target_mask_to_predictor:
      self.target_mask_embedding = nn.Embedding(2, self.d_model)
    # Enable gradient checkpointing for memory efficiency
    encoder_config = BertConfig(
      # Unnecessary vocab_size thing
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
  
  def _fuse_decoded_tokens(self, decoded_tokens, context_emb):
     if self.tokenization == 'remi':
        raise ValueError("Remi tokenization does not support fused tokens")
     else:
        decoded_projection = self.decoded_tokens_dropout(self.decoded_tokens_layer_norm(self.decoded_tokens_in(decoded_tokens)))
        fused_emb = self.fusion_dropout(self.fusion_layer_norm(self.fusion_layer(torch.cat([context_emb, decoded_projection], dim=-1))))
        return fused_emb

  def _fuse_fme(self, decoded_tokens, context_emb):
    pitch = Utils.get_pitch_sequence(decoded_tokens)
    duration = Utils.get_duration_sequence(decoded_tokens)

    pitch_emb = self.pitch_fme(pitch)
    duration_emb = self.duration_fme(duration)

    fused_fme = self.fuse_pitch_duration_layer(torch.cat([pitch_emb, duration_emb], dim=-1))
    fused_fme = self.fuse_fme_dropout(self.fuse_fme_layer_norm(self.fuse_fme_layer(torch.cat([context_emb, fused_fme], dim=-1))))

    return fused_fme

  def _fuse_custom_continuous_tokens(self, decoded_tokens, context_emb):
    continuous_tokens = Utils.get_continuous_tokens(decoded_tokens)
    continuous_tokens_projection = self.continuous_tokens_dropout(self.continuous_tokens_layer_norm(self.continuous_tokens_in(continuous_tokens)))
    fused_emb = self.fuse_custom_continuous_tokens_dropout(self.fuse_custom_continuous_tokens_layer_norm(self.fuse_custom_continuous_tokens_layer(torch.cat([context_emb, continuous_tokens_projection], dim=-1))))
    return fused_emb

  def embed(self, input_ids):
    # Context and target IDs should be already masked
    if self.tokenization == 'remi':
        context_emb = self.remi_in(input_ids)
    else:  # octuple
        # Embed tokens
        x = self.octuple_in(input_ids)  # [batch_size, seq_len, d_model]
        
        # Reshape to group every 8 tokens
        batch_size, seq_len, emb_dim = x.shape
        x = x.view(batch_size, seq_len // 8, 8 * emb_dim)
        
        # Project grouped embeddings
        context_emb = self.octuple_downsampling(x)
        context_emb = self.octuple_layer_norm(context_emb)
        context_emb = self.octuple_dropout(context_emb)

        decoded_tokens = Utils.decode_tokens(input_ids, self.vocab).to(self.device)

        if self.fuse_decoded_tokens:
            context_emb = self._fuse_decoded_tokens(decoded_tokens, context_emb)

        if self.fuse_fme:
            context_emb = self._fuse_fme(decoded_tokens, context_emb)
          
        if self.use_custom_continuous_tokens:
            context_emb = self._fuse_custom_continuous_tokens(decoded_tokens, context_emb)

        if self.add_onset_positional_encoding:
          context_emb = self.onset_positional_encoding(
            context_emb,
            global_timing=Utils.get_bar_sequence(decoded_tokens),
            modulo_timing=Utils.get_local_onset_sequence(decoded_tokens)
          )
        else:
          context_emb = context_emb + self.positional_encoding[:, :context_emb.size(1), :]

    return context_emb

  def encode_context(self, context_ids, context_mask=None):
    context_emb = self.embed(context_ids)
    
    out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    encoder_hidden = out.hidden_states[-1]

    if context_mask is not None:
      if self.tokenization == 'octuple':
        context_mask = context_mask[:, ::8]
      encoder_hidden[context_mask] = 0
    return encoder_hidden

  def forward(self, context_ids, target_ids=None, return_context_encoder_hidden=False, context_mask=None, target_mask=None):
    encoder_hidden = self.encode_context(context_ids, context_mask)

    if target_ids is not None:
        with torch.no_grad():
            target_emb = self.embed(target_ids)
            target_emb = target_emb + self.positional_encoding[:, :target_emb.size(1), :]
            out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
            target_encoder_hidden = out.last_hidden_state

        if self.tokenization == 'octuple':
            target_mask = target_mask[:, ::8]
        target_encoder_hidden[target_mask] = 0

        predictor_input = encoder_hidden + self.positional_encoding[:, :encoder_hidden.size(1), :]
        if self.pass_target_mask_to_predictor:
            latent_var = self.target_mask_embedding(target_mask.to(torch.int)) + self.positional_encoding[:, :target_mask.size(1), :]
            predictor_input = torch.cat([predictor_input, latent_var], dim=1)

        pred = self.predictor(inputs_embeds=predictor_input, output_hidden_states=True)
        pred_hidden = pred.last_hidden_state[:, encoder_hidden.size(1):]

        return pred_hidden, target_encoder_hidden, encoder_hidden
    return pred_hidden
    
  def vicreg_loss(self, context_hidden):
    all_vectors = context_hidden.view(-1, context_hidden.size(-1))
    
    N_total = all_vectors.size(1)
    perm = torch.randperm(N_total)
    z1 = all_vectors[perm[:N_total//2]]
    z2 = all_vectors[perm[N_total//2:]]
    
    N, D = z1.shape

    # Invariance (MSE)
    sim_loss = inv = F.mse_loss(z1, z2)

    # Combine for stats
    z = torch.cat([z1, z2], dim=0)
    z = z - z.mean(dim=0, keepdim=True)

    # Variance term: hinge on per-dim std >= var_target
    var_target = 1.0
    std = torch.sqrt(z.var(dim=0) + 1e-5)
    var_loss = torch.mean(F.relu(var_target - std))

    # Covariance term: off-diagonals of correlation matrix
    cov = (z.T @ z) / (2 * N - 1)
    d = torch.diag(cov)
    inv_s = torch.rsqrt(d + 1e-5)
    corr = inv_s[:, None] * cov * inv_s[None, :]
    off_diag = ~torch.eye(D, device=z.device, dtype=torch.bool)
    cov_loss = corr[off_diag].pow(2).sum() / D
    
    # Combine losses with weights
    total_loss = (
        self.vicreg_sim_weight * inv + 
        self.vicreg_var_weight * var_loss + 
        self.vicreg_cov_weight * cov_loss
    )
    
    return total_loss, sim_loss, var_loss, cov_loss

  def get_loss(self, batch, fold='train'):
    pred, target, context_hidden = self(
      batch['context_ids'],
      batch['target_ids'],
      return_context_encoder_hidden=True,
      context_mask=batch.get('context_mask'),
      target_mask=batch.get('target_mask'))

    # TODO: Mask preds to only keep the positions of masked tokens
    
    # Original JEPA loss
    jepa_loss = self.loss_fn(pred, target)
    
    if self.use_vicreg:
      vicreg_total, vic_sim, vic_var, vic_cov = self.vicreg_loss(context_hidden)

      self.log(f'{fold}_vicreg_loss', vicreg_total, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      # Dynamic loss weighting
      with torch.no_grad():
          loss_ratio = vicreg_total / jepa_loss
          scale = self.vicreg_loss_ratio / loss_ratio if loss_ratio > self.vicreg_loss_ratio else 1.0
          
      # Apply dynamic scaling
      vicreg_total = vicreg_total * scale
      
      # Log VicReg components
      self.log(f'{fold}_vicreg_sim', vic_sim, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      self.log(f'{fold}_vicreg_var', vic_var, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      self.log(f'{fold}_vicreg_cov', vic_cov, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      
      # Combine losses
      total_loss = jepa_loss + vicreg_total
      self.log(f'{fold}_jepa_loss', jepa_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
      
      return total_loss
    
    return jepa_loss

  def on_train_batch_start(self, batch, batch_idx):
    current_context_ratio = self.trainer.datamodule.collator.ratio_context_step()
    self.log('context_ratio', current_context_ratio, on_step=True, logger=True, sync_dist=True)
  
  def training_step(self, batch, batch_idx):
    loss = self.get_loss(batch, fold='train')

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
    loss = self.get_loss(batch, fold='val')
    
    # Get encoder outputs
    # if self.tokenization == 'remi':
    #     context_emb = self.remi_in(batch['context_ids'])
    #     context_out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    #     context_hidden = context_out.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
    # elif self.tokenization == 'octuple':
    #     context_emb = self.octuple_in(batch['context_ids'])

    #     context_out = self.context_encoder(inputs_embeds=context_emb, output_hidden_states=True)
    #     context_hidden = context_out.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
    
    # with torch.no_grad():
    #     target_emb = self.remi_in(batch['target_ids'])
    #     target_out = self.target_encoder(inputs_embeds=target_emb, output_hidden_states=True)
    #     target_hidden = target_out.hidden_states[-1]
        
    #     # Calculate representation collapse metrics
        
    #     # 1. Cosine similarity (higher value indicates more collapse)
    #     cos_sim = F.cosine_similarity(
    #         context_hidden.view(-1, context_hidden.size(-1)),
    #         target_hidden.view(-1, target_hidden.size(-1)),
    #         dim=1
    #     ).mean()
        
    #     # 2. Mean squared difference of normalized representations (lower value indicates more collapse)
    #     context_norm = F.normalize(context_hidden, p=2, dim=-1)
    #     target_norm = F.normalize(target_hidden, p=2, dim=-1)
    #     mse_diff = torch.mean((context_norm - target_norm) ** 2)
        
    #     # 3. Variance of representations across batch
    #     # Reshape to [batch_size * seq_len, hidden_dim]
    #     context_flat = context_hidden.view(-1, context_hidden.size(-1))
    #     target_flat = target_hidden.view(-1, target_hidden.size(-1))
        
    #     # Calculate variance for each feature across batch
    #     context_var = torch.var(context_flat, dim=0).mean()  # Mean variance across features
    #     target_var = torch.var(target_flat, dim=0).mean()
        
    #     # Log all metrics
    #     self.log('val_cosine_similarity', cos_sim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.log('val_mse_diff', mse_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.log('val_context_variance', context_var, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.log('val_target_variance', target_var, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    #     # Also log the original norms
    #     context_norm = torch.norm(context_hidden, dim=-1).mean()
    #     target_norm = torch.norm(target_hidden, dim=-1).mean()
    #     self.log('val_context_encoder_norm', context_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    #     self.log('val_target_encoder_norm', target_norm, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    return self.get_loss(batch, fold='test')
        
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



class SymJEPAPooler(nn.Module):
  """
  Pools using 4 different strategies:
  - max
  - mean
  - CLS
  - Attentive

  Then fuses all of them together using a linear layer
  """
  def __init__(self,
               d_model=512,
               **kwargs):
    super().__init__()


    self.attn_score = nn.Linear(d_model, 1)

    self.fuse = nn.Linear(4 * d_model, d_model)

  def forward(self, vectors):
    cls_token = vectors[:, 0, :]
    max_token, _ = vectors.max(dim=1)
    mean_token = vectors.mean(dim=1)
    attentive = self.attentive_pooling(vectors)

    return self.fuse(torch.cat([cls_token, max_token, mean_token, attentive], dim=1))

  def attentive_pooling(self, vectors):
    attn = F.softmax(self.attn_score(vectors).squeeze(-1), dim=-1)
    return (attn @ vectors).sum(dim=1)
    
