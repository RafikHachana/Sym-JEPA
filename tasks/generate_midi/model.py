import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from model import SymJEPA
from octuple_tokenizer import OctupleTokenizer, OctupleVocab, get_max_vector, get_separate_vocabs
from masking import _get_instrument_id
from tqdm import tqdm


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
            position_embedding_type='absolute',
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


    def forward(self, input_ids, sampling_prob=0.0):
        self.jepa.eval()
        with torch.no_grad():
            encoder_hidden = self.jepa.encode_context(input_ids)

        # Prepare causal mask
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len // 8, seq_len // 8, device=input_ids.device),
            diagonal=0
        ).unsqueeze(0).repeat(input_ids.shape[0], 1, 1)

        input_ids_model = input_ids

        # Efficient vectorized scheduled sampling
        if sampling_prob > 0.0:
            with torch.no_grad():
                embeds = self.jepa.embed(input_ids)
                teacher_forced_out = self.generator(
                    inputs_embeds=embeds,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=None,
                    attention_mask=causal_mask,
                ).last_hidden_state

                teacher_forced_logits = []
                for i, out_layer in enumerate(self.out_layers):
                    teacher_forced_logits.append(out_layer(teacher_forced_out))


            # Sample from teacher-forced logits
            sampled_tokens = []
            for l in teacher_forced_logits:
                # print("Logit shape: ", l.shape)
                # sampled_tokens.append(torch.multinomial(
                #     torch.softmax(l, dim=-1),
                #     num_samples=1
                # ).squeeze(-1))

                sampled_tokens.append(torch.argmax(l, dim=-1))

            all_sampled_tokens = []
            for i in range(sampled_tokens[0].shape[1]):
                for j in range(8):
                    all_sampled_tokens.append(sampled_tokens[j][:, i])

            for i in range(len(all_sampled_tokens)):
                all_sampled_tokens[i] = torch.tensor(self.octuple_vocab.encode(self.vocabs[i%8].decode(all_sampled_tokens[i].cpu())), device=input_ids.device).unsqueeze(-1)

                # print("Sampled token: ", all_sampled_tokens[i].shape)

            sampled_tokens = torch.cat(all_sampled_tokens, dim=1)


            # print("Sampled tokens: ", sampled_tokens.shape)

            # Don't replace the first token (usually BOS)
            sampling_mask = (torch.rand_like(input_ids[:, 8:].float()) < sampling_prob)
            sampling_mask[:, 0] = False  # always use BOS

            # Apply scheduled sampling
            input_ids_model = torch.where(sampling_mask, sampled_tokens[:, :-8], input_ids[:, 8:])

            input_ids_model = torch.cat((input_ids[:, :8], input_ids_model), dim=1)

        embeds = self.jepa.embed(input_ids_model)
        # Forward pass using possibly modified input_ids_model
        out = self.generator(
            inputs_embeds=embeds,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=None,
            attention_mask=causal_mask,
        ).last_hidden_state

        # Output separate logits for each token type
        logits = []
        for i, out_layer in enumerate(self.out_layers):
            logits.append(out_layer(out))

        return logits


    def get_loss(self, logits, labels, fold='train'):
        loss = 0

        avg_accuracy = 0

        for i, (logit, vocab, token_type) in enumerate(zip(logits, self.vocabs, self.token_types)):
            relevant_labels_original_vocab = labels[:, i::8].reshape(-1)

            relevant_labels = torch.tensor(vocab.encode(self.octuple_vocab.decode(relevant_labels_original_vocab.cpu())), device=logit.device)


            token_loss = self.loss(logit.reshape(-1, len(vocab)), relevant_labels)
            loss += token_loss
            self.log(f'{fold}_{token_type}_loss', token_loss)

            # Log the accuracy
            accuracy = (logit.reshape(-1, len(vocab)).argmax(dim=-1) == relevant_labels).float().mean()
            self.log(f'{fold}_{token_type}_accuracy', accuracy)
            avg_accuracy += accuracy

            # Log the last token accuracy?
        
        return loss, avg_accuracy / len(self.token_types)

    def training_step(self, batch, batch_idx):
        sampling_prob = 0.0
        if batch_idx > 200:
            sampling_prob = 0.5
        input_ids = batch['input_ids']  # Remove the last token for prediction
        labels = batch['input_ids'][:, 8:-8]  # Shift the input for the
        logits = self(input_ids[:, :-16], sampling_prob=sampling_prob)
        loss, accuracy = self.get_loss(logits, labels, fold='train')
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']  # Remove the last token for prediction
        labels = batch['input_ids'][:, 8:-8]  # Shift the input for the
        logits = self(input_ids[:, :-16], sampling_prob=0.1)
        loss, accuracy = self.get_loss(logits, labels, fold='val')
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)

        # Try accuracy without teacher-forcing (only on a few batches because it takes time)
        if batch_idx % 400 == 0:
            decoded = self.decode(self.jepa.encode_context(input_ids))
            self.log('val_autoreg_acc', (decoded == input_ids).float().mean())
        return loss
    
    def decode(self, encoder_hidden, initial_tokens=None, use_tqdm=False):
        octuple_vocab = OctupleVocab()
        if initial_tokens is not None:
            generated_ids = initial_tokens.to(self.device)
        else:
            bos = OctupleTokenizer.get_bos_eos_tokens()[0]
            bos_encoded = torch.tensor([octuple_vocab.encode([bos])], device=self.device).repeat(encoder_hidden.shape[0], 8)
            generated_ids = bos_encoded
        # print("Autoregressively generating OctupleMIDI tokens...")
        # We can either count on the decoder to generate the EOS token or we can manually add it

        music_format = "🎼 {desc}: {percentage:3.0f}%|{bar:100}|[{elapsed}<{remaining}]"

        it = range(encoder_hidden.shape[1]-1)
        if use_tqdm:
            it = tqdm(it, colour='cyan', desc='Generating OctupleMIDI tokens', bar_format=music_format)

        for _ in it: 

            generated_embeds = self.jepa.embed(generated_ids)
            out = self.generator(
                inputs_embeds=generated_embeds,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=None,  # Assuming no attention mask for the encoder
            ).last_hidden_state

            # Output separate logits for each token type
            for i, (out_layer, vocab) in enumerate(zip(self.out_layers, self.vocabs)):
                logit = out_layer(out)

                # print("Logit shape: ", logit.shape)

                predicted_id_in_sub_vocab = logit[:, -1:, :].argmax(dim=-1)

                # print(predicted_id_in_sub_vocab.shape)


                predicted_id = torch.tensor(octuple_vocab.encode(vocab.decode(predicted_id_in_sub_vocab.cpu())), device=self.device)

                # print("Predicted id shape: ", predicted_id.shape)

                generated_ids = torch.cat((generated_ids, predicted_id.unsqueeze(-1)), dim=1)
            # print(generated_ids)
        
        return generated_ids

    def generate_instrument(self, input_ids, instrument_id, relative_note_density=0.1):
        input_ids = input_ids.unsqueeze(0)
        n_mask_tokens = int(input_ids.shape[1] * relative_note_density) // 8

        # Sample positions for the masks, with replacement (not including the first and last 8 tokens, they are BOS and EOS)
        mask_positions = sorted((torch.randint(1, input_ids.shape[1] // 8 - 1, (n_mask_tokens,), device=input_ids.device)*8).numpy().tolist())

        # Create a mask for the positions
        mask_token_id = self.octuple_vocab.encode([OctupleTokenizer.get_mask_token()])[0]
        masked_token = [mask_token_id] * 8


        masked_input_ids = []

        for i in range(len(mask_positions)):
            if i == 0:
                masked_input_ids.append(input_ids[0, :mask_positions[i]].tolist())
            else:
                masked_input_ids.append(input_ids[0, mask_positions[i-1]:mask_positions[i]].tolist())
            masked_input_ids.append(masked_token)

        masked_input_ids.append(input_ids[0, mask_positions[-1]:].tolist())

        # print(masked_input_ids)
        masked_input_ids = torch.cat([torch.tensor(t) for t in masked_input_ids], dim=0).unsqueeze(0)

        target_mask = torch.ones_like(masked_input_ids)
        target_mask[masked_input_ids == mask_token_id] = 0


        minimized_target_mask = target_mask[:, ::8]

        latent_var_ids = torch.tensor([_get_instrument_id(instrument_id)] * (masked_input_ids.shape[1] // 8), device=masked_input_ids.device).unsqueeze(0)

        encoder_hidden = self.jepa.encode_context(masked_input_ids.long().to(self.device))


        predictor_output = self.jepa.predict(encoder_hidden, latent_var_ids.long().to(self.device), minimized_target_mask.bool().to(self.device))


        decoder_input = encoder_hidden.clone()

        decoder_input[minimized_target_mask == 0] = predictor_output

        initial_ids = input_ids[:, :mask_positions[0]]

        # Generate the instrument
        instrument_ids = self.decode(self.jepa.encode_context(input_ids.long().to(self.device)), initial_tokens=initial_ids)
        # instrument_ids = self.decode(decoder_input, initial_tokens=initial_ids)

        return instrument_ids


    def load_jepa(self, ckpt_path):
        self.jepa.load_state_dict(torch.load(ckpt_path)['state_dict'])
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    
