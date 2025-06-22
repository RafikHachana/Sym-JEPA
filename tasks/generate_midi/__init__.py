import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from model import SymJEPA
from octuple_tokenizer import OctupleTokenizer, OctupleVocab, get_max_vector, get_separate_vocabs


class MusicDecoder(pl.LightningModule):
    def __init__(self, 
                    max_len=2048,
                    lr=1e-3,
                    d_model=512,
                    encoder_layers=8,
                    num_attention_heads=8,
                    intermediate_size=2048,
                    **kwargs):
        super().__init__()


        self.max_len = max_len
        
        # Initialize only the encoder
        self.jepa = SymJEPA(tokenization='octuple', pass_target_mask_to_predictor=True)
        
        self.jepa.eval()

        self.generator = BertModel(config=BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=d_model,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=1024,
            position_embedding_type='relative_key_query',
            is_decoder=True,
            add_cross_attention=True
        ))
        
        # Get the vocabulary sizes for each token type
        self.vocab_sizes = get_max_vector()

        self.vocabs = get_separate_vocabs()

        self.octuple_vocab = OctupleVocab()

        self.token_types = ['bar', 'note', 'inst', 'pitch', 'duration', 'velocity', 'ts', 'tempo']
        
        # Create separate output layers for each token type
        self.out_layers = nn.ModuleList([
            nn.Linear(d_model, len(vocab), bias=False) 
            for vocab in self.vocabs
        ])

        self.lr = lr
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()


    def forward(self, input_ids):
        self.jepa.eval()
        with torch.no_grad():
            encoder_hidden = self.jepa.encode_context(input_ids)
        
        embeds = self.jepa.embed(input_ids)

        out = self.generator(
            inputs_embeds=embeds,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=None,  # Assuming no attention mask for the encoder
            attention_mask=torch.triu(torch.ones(input_ids.shape[1]//8, input_ids.shape[1]//8, device=input_ids.device), diagonal=0).unsqueeze(0).repeat(input_ids.shape[0], 1, 1),  # Causal mask for decoder
        ).last_hidden_state


        # Output separate logits for each token type
        logits = []
        for i, out_layer in enumerate(self.out_layers):
            logits.append(out_layer(out))
        
        return logits

    def get_loss(self, logits, labels):
        loss = 0

        for i, (logit, vocab, token_type) in enumerate(zip(logits, self.vocabs, self.token_types)):
            relevant_labels_original_vocab = labels[:, i::8].reshape(-1)

            relevant_labels = torch.tensor(vocab.encode(self.octuple_vocab.decode(relevant_labels_original_vocab.cpu())), device=logit.device)


            token_loss = self.loss(logit.reshape(-1, len(vocab)), relevant_labels)
            loss += token_loss
            self.log(f'{token_type}_loss', token_loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']  # Remove the last token for prediction
        labels = batch['input_ids'][:, 8:-8]  # Shift the input for the
        logits = self(input_ids[:, :-16])
        loss = self.get_loss(logits, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']  # Remove the last token for prediction
        labels = batch['input_ids'][:, 8:-8]  # Shift the input for the
        logits = self(input_ids[:, :-16])
        loss = self.get_loss(logits, labels)
        self.log('val_loss', loss)
        return loss
    
    def generate(self, encoder_hidden):
        
        tokenizer = OctupleTokenizer()
        bos = tokenizer.get_bos_eos_tokens()[0]
        bos_encoded = torch.tensor([[tokenizer.get_vocab().encode(bos)]], device=self.device).repeat(encoder_hidden.shape[0], 8)

        generated_ids = bos_encoded
        for _ in range(self.max_len - 1): 
            new_seq = self.generator(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=None,  # Assuming no attention mask for the encoder
            )

            generated_ids = torch.cat((generated_ids, new_seq.logits[:, -1:, :].argmax(dim=-1)), dim=1)
        
        return generated_ids

    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    