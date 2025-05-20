import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import math
import os
import pickle
import argparse
from glob import glob
import logging
import numpy as np
import json
import traceback
import hashlib
from input_representation import RemiTokenizer
from vocab import RemiVocab
from constants import (
  BAR_KEY, POSITION_KEY
)
from tqdm import tqdm


CACHE_PATH = os.getenv('CACHE_PATH', os.getenv('SCRATCH', os.getenv('TMPDIR', './temp')))

def get_md5(file_path):
  return hashlib.md5(open(file_path, 'rb').read()).hexdigest()

class MidiDataModule(pl.LightningDataModule):
  def __init__(self, 
               files,
               max_len,
               batch_size=32, 
               num_workers=4,
               pin_memory=True, 
               train_val_test_split=(0.95, 0.1, 0.05),
               jepa_context_ratio=0.75,
               use_mask_padding=False,
               masking_mode='contiguous',
               masking_probability=0.25,
               segment_size_ratio=0.1,
               num_segments=3,
               tokenization='remi',
               genre_map='metadata/midi_genre_map.json',
               skip_unknown_genres=False,
               skip_unknown_styles=False,
               generate_melody_completion_pairs=False,
               **kwargs):
    super().__init__()
    self.batch_size = batch_size
    self.pin_memory = pin_memory
    self.num_workers = num_workers
    self.files = files
    self.train_val_test_split = train_val_test_split
    self.max_len = max_len
    self.jepa_context_ratio = jepa_context_ratio
    self.use_mask_padding = use_mask_padding
    self.masking_mode = masking_mode
    self.masking_probability = masking_probability
    self.segment_size_ratio = segment_size_ratio
    self.num_segments = num_segments
    self.kwargs = kwargs
    self.tokenization = tokenization
    self.genre_map = genre_map
    self.skip_unknown_genres = skip_unknown_genres
    self.skip_unknown_styles = skip_unknown_styles
    self.generate_melody_completion_pairs = generate_melody_completion_pairs
    if tokenization == 'remi':
        from input_representation import RemiTokenizer
        self.tokenizer_class = RemiTokenizer
    elif tokenization == 'octuple':
        from octuple_tokenizer import OctupleTokenizer
        self.tokenizer_class = OctupleTokenizer
    else:
        raise ValueError(f"Unknown tokenization method: {tokenization}")
    self.vocab = self.tokenizer_class.get_vocab()
    self.eos_bos_tokens = self.tokenizer_class.get_bos_eos_tokens()
    self.mask_token = self.tokenizer_class.get_mask_token()
    self.unk_token = self.tokenizer_class.get_unk_token()
    self.pad_token = self.tokenizer_class.get_pad_token()

    self.setup_done = False

  def setup(self, stage=None):
    if self.setup_done:
      return
    # n_train = int(self.train_val_test_split[0] * len(self.files))
    n_valid = int(self.train_val_test_split[1] * len(self.files))
    n_test = int(self.train_val_test_split[2] * len(self.files))
    train_files = self.files[n_test+n_valid:]
    valid_files = self.files[n_test:n_test+n_valid]
    test_files = self.files[:n_test]

    self.train_ds = MidiDataset(train_files, self.max_len, 
      tokenization=self.tokenization,
      tokenizer_class=self.tokenizer_class,
      skip_unknown_genres=self.skip_unknown_genres,
      skip_unknown_styles=self.skip_unknown_styles,
      **self.kwargs
    )
    self.valid_ds = MidiDataset(valid_files, self.max_len, 
      tokenization=self.tokenization,
      tokenizer_class=self.tokenizer_class,
      skip_unknown_genres=self.skip_unknown_genres,
      skip_unknown_styles=self.skip_unknown_styles,
      **self.kwargs
    )
    self.test_ds = MidiDataset(test_files, self.max_len, 
      tokenization=self.tokenization,
      tokenizer_class=self.tokenizer_class,
      skip_unknown_genres=self.skip_unknown_genres,
      skip_unknown_styles=self.skip_unknown_styles,
      **self.kwargs
    )

    print(f"Train dataset size: {len(self.train_ds)}")
    print(f"Valid dataset size: {len(self.valid_ds)}")
    print(f"Test dataset size: {len(self.test_ds)}")

    self.collator = SeqCollator(
        pad_token=self.vocab.to_i(self.pad_token),
        mask_token=self.vocab.to_i(self.mask_token),
        context_size=self.max_len,
        jepa_context_ratio=self.jepa_context_ratio,
        use_mask_padding=self.use_mask_padding,
        masking_mode=self.masking_mode,
        masking_probability=self.masking_probability,
        segment_size_ratio=self.segment_size_ratio,
        num_segments=self.num_segments,
        generate_melody_completion_pairs=self.generate_melody_completion_pairs
    )

    self.skipped_files = self.train_ds.skipped_files + self.valid_ds.skipped_files + self.test_ds.skipped_files

    self.setup_done = True


  def train_dataloader(self):
    return DataLoader(self.train_ds, 
                      collate_fn=self.collator, 
                      batch_size=self.batch_size, 
                      pin_memory=self.pin_memory, 
                      shuffle=True,
                      num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.valid_ds, 
                      collate_fn=self.collator, 
                      batch_size=self.batch_size, 
                      pin_memory=self.pin_memory, 
                      num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_ds, 
                      collate_fn=self.collator, 
                      batch_size=self.batch_size, 
                      pin_memory=self.pin_memory, 
                      num_workers=self.num_workers)


def _get_split(files, worker_info):
  if worker_info:
    n_workers = worker_info.num_workers
    worker_id = worker_info.id

    per_worker = math.ceil(len(files) / n_workers)
    start_idx = per_worker*worker_id
    end_idx = start_idx + per_worker

    split = files[start_idx:end_idx]
  else:
    split = files
  return split


class SeqCollator:
  def __init__(self, pad_token=0, mask_token=None, context_size=2048, 
               jepa_context_ratio=0.75, use_mask_padding=False,
               masking_mode='contiguous',
               masking_probability=0.25,
               segment_size_ratio=0.1,
               num_segments=3,
               tokenization='remi',
               generate_melody_completion_pairs=False):
    self.pad_token = pad_token
    self.mask_token = mask_token
    self.context_size = context_size
    self.jepa_context_ratio = jepa_context_ratio
    self.use_mask_padding = use_mask_padding
    self.masking_mode = masking_mode
    self.masking_probability = masking_probability
    self.segment_size_ratio = segment_size_ratio
    self.num_segments = num_segments
    self.tokenization = tokenization
    self.generate_melody_completion_pairs = generate_melody_completion_pairs

    self.mask_step = 1
    if tokenization == 'octuple':
      self.mask_step = 8

  def create_masks(self, seq_length, mask_step=1):
    # IMPORTANT: Only the contiguous masking works with the octuple tokenization
    if self.masking_mode == 'contiguous':
      # Original contiguous masking
      mask_start = int(seq_length // mask_step * self.jepa_context_ratio) * mask_step
      assert mask_start % mask_step == 0, f"mask_start {mask_start} is not divisible by mask_step {mask_step}"
      mask = torch.zeros(seq_length, dtype=torch.bool)
      mask[mask_start:] = True

    elif self.masking_mode == 'random_contiguous':
      # Random contiguous masking
      mask_start_max = int(seq_length // mask_step * self.jepa_context_ratio) * mask_step
      assert mask_start_max % mask_step == 0, f"mask_start_max {mask_start_max} is not divisible by mask_step {mask_step}"
      mask_start = torch.randint(0, mask_start_max, (1,))
      mask = torch.zeros(seq_length, dtype=torch.bool)
      mask[mask_start:] = True
      
    elif self.masking_mode == 'random':
      # Random token masking
      mask = torch.rand(seq_length) < self.masking_probability
      
    elif self.masking_mode == 'segments':
      # Non-overlapping segments
      mask = torch.zeros(seq_length, dtype=torch.bool)
      segment_size = int(seq_length * self.segment_size_ratio)
      valid_starts = list(range(0, seq_length - segment_size))
      
      # Randomly select start positions for segments
      if len(valid_starts) >= self.num_segments:
        start_positions = torch.tensor(
            sorted(np.random.choice(valid_starts, self.num_segments, replace=False))
        )
        
        # Create masks for each segment
        for start in start_positions:
          mask[start:start + segment_size] = True
      else:
        # Fallback to random masking if sequence is too short
        mask = torch.rand(seq_length) < self.masking_probability
    
    return mask

  def __call__(self, features):
    batch = {}
    
    xs_list = [feature['input_ids'] for feature in features]
    xs = pad_sequence(xs_list, batch_first=True, padding_value=self.pad_token)

    if self.context_size > 0:
      max_len = min(self.context_size, xs.size(1))  # Use actual sequence length
    else:
      max_len = xs.size(1)

    if self.use_mask_padding:
      batch_size = xs.size(0)
      # Create masks using actual sequence length
      masks = [self.create_masks(xs.size(1), mask_step=self.mask_step) for _ in range(batch_size)]
      masks = torch.stack(masks)
      
      # Create masked versions for context and target
      context = xs.clone()
      target = xs.clone()
      
      # Apply masks
      context[masks] = self.mask_token  # Mask target tokens in context
      target[~masks] = self.mask_token  # Mask context tokens in target

      batch['context_mask'] = masks
      batch['target_mask'] = ~masks
      
    else:
      # Original padding method
      jepa_context_size = int(xs.size(1) * self.jepa_context_ratio)
      context = xs[:, :jepa_context_size]
      target = xs[:, jepa_context_size:]
      
      # Pad to max_len
      if context.size(1) < max_len:
          pad_size = max_len - context.size(1)
          context = torch.nn.functional.pad(context, (0, pad_size), value=self.pad_token)
      if target.size(1) < max_len:
          pad_size = max_len - target.size(1)
          target = torch.nn.functional.pad(target, (0, pad_size), value=self.pad_token)

    batch['context_ids'] = context[:, :max_len]  # Ensure we don't exceed max_len
    batch['target_ids'] = target[:, :max_len]
    if all(feature['genre_id'] is not None for feature in features):
      batch['genre_id'] = torch.tensor([feature['genre_id'] for feature in features], dtype=torch.long)
    else:
      batch['genre_id'] = None

    if all(feature['style_id'] is not None for feature in features):
      batch['style_id'] = torch.tensor([feature['style_id'] for feature in features], dtype=torch.long)
    else:
      batch['style_id'] = None

    batch['input_ids'] = xs

    if self.generate_melody_completion_pairs:
      batch['melody_completion_start'], batch['melody_completion_end'], batch['melody_completion_match'] = self._generate_melody_completion_pairs(xs_list)
    
    return batch

  def _generate_melody_completion_pairs(self, xs_list):
    melody_completion_start = []
    melody_completion_end = []
    melody_completion_match = []

    batch_size = len(xs_list)

    for i in range(batch_size):
      start_ind = torch.randint(0, batch_size, (1,))
      first_sample = xs_list[start_ind]
      melody_completion_start.append(first_sample[:first_sample.size(0)//2])

      positive_sample = torch.rand(1).item() < 0.5

      if positive_sample:
        # Generate a positive sample
        
        melody_completion_end.append(first_sample[first_sample.size(0)//2:])
        melody_completion_match.append(1)
      else:
        # Generate a negative sample
        second_sample = xs_list[torch.randint(0, batch_size, (1,))]

        melody_completion_end.append(second_sample[second_sample.size(0)//2:])
        melody_completion_match.append(0)

    return (
      pad_sequence(melody_completion_start, batch_first=True, padding_value=self.pad_token),
      pad_sequence(melody_completion_end, batch_first=True, padding_value=self.pad_token),
      torch.tensor(melody_completion_match, dtype=torch.long)
    )





class MidiDataset(torch.utils.data.Dataset):
  def __init__(self, 
               midi_files, 
               max_len, 
               tokenization='remi',
               group_bars=False, 
               max_bars=512, 
               max_positions=512,
               max_bars_per_context=-1,
               max_contexts_per_file=-1,
               bar_token_mask=None,
               bar_token_idx=2,
               use_cache=True,
               print_errors=False,
               tokenizer_class=RemiTokenizer,
               use_mask_padding=False,
               genre_map_path='metadata/midi_genre_map.json',
               skip_unknown_genres=False,
               skip_unknown_styles=False,
               generate_melody_completion_pairs=False,
               sample_count_per_sequence=1):
    self.files = midi_files
    self.group_bars = group_bars
    self.max_len = max_len
    self.max_bars = max_bars
    self.max_positions = max_positions
    self.max_bars_per_context = max_bars_per_context
    self.max_contexts_per_file = max_contexts_per_file
    self.use_cache = use_cache
    self.print_errors = print_errors
    self.use_mask_padding = use_mask_padding
    self.tokenization = tokenization
    self.skip_unknown_genres = skip_unknown_genres
    self.skip_unknown_styles = skip_unknown_styles
    self.tokenizer_class = tokenizer_class
    self.sample_count_per_sequence = sample_count_per_sequence
    with open(genre_map_path, 'r') as f:
      self.genre_map = json.load(f)

    self.all_genres = set(sum(self.genre_map['topmagd'].values(), []))
    self.all_styles = set(sum(self.genre_map['masd'].values(), []))

    self.genre_to_idx = {genre: i for i, genre in enumerate(self.all_genres)}
    self.idx_to_genre = {i: genre for i, genre in enumerate(self.all_genres)}

    self.style_to_idx = {style: i for i, style in enumerate(self.all_styles)}
    self.idx_to_style = {i: style for i, style in enumerate(self.all_styles)}

    self.genre_counts = torch.zeros(len(self.all_genres))
    self.style_counts = torch.zeros(len(self.all_styles))

    self.vocab = self.tokenizer_class.get_vocab()

    self.bar_token_mask = bar_token_mask
    self.bar_token_idx = bar_token_idx

    if CACHE_PATH:
      self.cache_path = os.path.join(CACHE_PATH, self.tokenizer_class.version())
      os.makedirs(self.cache_path, exist_ok=True)
    else:
      self.cache_path = None
    
    # Pre-process all files and store their data
    self.data = []

    self.skipped_files = []
    for file in tqdm(self.files):
      try:
        current_file = self.load_file(file)
        events = current_file['events']


        # Identify start of bars
        if self.tokenization == 'remi':
          bars, bar_ids = self.get_bars(events, include_ids=True)
          if len(bars) > self.max_bars:
            if self.print_errors:
              print(f"WARNING: REMI sequence has more than {self.max_bars} bars: {len(bars)} event bars.")
            continue

          # Identify positions
          position_ids = self.get_positions(events)
          max_pos = position_ids.max()
          if max_pos > self.max_positions:
            if self.print_errors:
              print(f"WARNING: REMI sequence has more than {self.max_positions} positions: {max_pos.item()} positions found")
            continue

          # Mask bar tokens if required
          if self.bar_token_mask is not None and self.max_bars_per_context > 0:
            events = self.mask_bar_tokens(events, bar_token_mask=self.bar_token_mask)
        
        # Encode tokens with appropriate vocabulary
        event_ids = torch.tensor(self.vocab.encode(events), dtype=torch.long)

        bos, eos = self.get_bos_eos_events()
        zero = torch.tensor([0], dtype=torch.int)

        if self.max_bars_per_context and self.max_bars_per_context > 0:
          # Find all indices where a new context starts based on number of bars per context
          starts = [bars[i] for i in range(0, len(bars), self.max_bars_per_context)]
          # Convert starts to ranges
          contexts = list(zip(starts[:-1], starts[1:])) + [(starts[-1], len(event_ids))]
        else:
          event_ids = torch.cat([bos, event_ids, eos])
          # bar_ids = torch.cat([zero, bar_ids, zero])
          # position_ids = torch.cat([zero, position_ids, zero])

          if self.max_len > 0:
            starts = list(range(0, len(event_ids), self.max_len+1))
            if len(starts) > 1:
              contexts = [(start, start + self.max_len) for start in starts[:-1]] + [(len(event_ids) - self.max_len, len(event_ids))]
            elif len(starts) > 0:
              contexts = [(starts[0], self.max_len)]
          else:
            contexts = [(0, len(event_ids))]

        if self.max_contexts_per_file and self.max_contexts_per_file > 0:
          contexts = contexts[:self.max_contexts_per_file]

        for start, end in contexts:
          # Add <bos> and <eos> to each context if contexts are limited to a certain number of bars
          if self.max_bars_per_context and self.max_bars_per_context > 0:
            src = torch.cat([bos, event_ids[start:end], eos])
            # b_ids = torch.cat([zero, bar_ids[start:end], zero])
            # p_ids = torch.cat([zero, position_ids[start:end], zero])
          else:
            src = event_ids[start:end]
            # b_ids = bar_ids[start:end]
            # p_ids = position_ids[start:end]

          if self.max_len > 0:
            src = src[:self.max_len + 1]

          file_id = os.path.basename(file).split('.')[0]

          # Check if file_id is a valid MD5 checksum
          if not len(file_id) == 32 or not all(c in '0123456789abcdef' for c in file_id.lower()):
            # print(f"WARNING: File ID {file_id} is not a valid MD5 checksum")
            file_id = get_md5(file)

          genre = self.genre_map['topmagd'].get(file_id, [None])[0]
          style = self.genre_map['masd'].get(file_id, [None])[0]
          
          if genre is None and self.skip_unknown_genres:
            continue

          if style is None and self.skip_unknown_styles:
            continue

          if genre is not None:
            self.genre_counts[self.genre_to_idx[genre]] += 1

          if style is not None:
            self.style_counts[self.style_to_idx[style]] += 1

          for _ in range(self.sample_count_per_sequence):
            self.data.append({
              'input_ids': src,
              'file': os.path.basename(file),
              # 'bar_ids': b_ids,
              'file_id': file_id,
              # 'position_ids': p_ids,
              'genre_id': self.genre_to_idx[genre] if genre else None,
              'style_id': self.style_to_idx[style] if style else None,
            })

      except ValueError as err:
        # traceback.print_exc()
        if self.print_errors:
          print(f"Error loading file {file}: {err}")       
        self.skipped_files.append(file)
        continue

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def get_bars(self, events, include_ids=False):
    # Seems like we have the bar tokens at this point, so it should be ok to just get them from the sequence
    bars = [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]
    
    if include_ids:
      bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
      bar_ids = torch.cumsum(bar_ids, dim=0)

      return bars, bar_ids
    else:
      return bars

  def get_positions(self, events):
    events = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in event else event for event in events]
    position_events = [event if f"{POSITION_KEY}_" in event else None for event in events]

    positions = [int(pos.split('_')[-1]) if pos is not None else None for pos in position_events]

    if positions[0] is None:
      positions[0] = 0
    for i in range(1, len(positions)):
      if positions[i] is None:
        positions[i] = positions[i-1]
    positions = torch.tensor(positions, dtype=torch.int)

    return positions

  def mask_bar_tokens(self, events, bar_token_mask='<mask>'):
    events = [bar_token_mask if f'{BAR_KEY}_' in token else token for token in events]
    return events
  
  def get_bos_eos_events(self):
    tuple_size = 8 if self.tokenization == 'octuple' else 1
    bos_event = torch.tensor(self.vocab.encode([self.tokenizer_class.get_bos_eos_tokens()[0]]*tuple_size), dtype=torch.long)
    eos_event = torch.tensor(self.vocab.encode([self.tokenizer_class.get_bos_eos_tokens()[1]]*tuple_size), dtype=torch.long)
    return bos_event, eos_event

  def load_file(self, file):
    """
    It seems that here we:
      - load the file
      - returns a dictionary with the REMI/OctupleMIDI tokens
    """
    name = os.path.basename(file)
    if self.cache_path and self.use_cache:
        cache_file = os.path.join(self.cache_path, f"{self.tokenization}_{name}")

    try:
        sample = pickle.load(open(cache_file, 'rb'))
    except Exception:
        try:
            tokenizer = self.tokenizer_class(file, strict=True)
            events = tokenizer.get_events()
        except Exception as err:
            # traceback.print_exc()
            raise ValueError(f'Unable to load file {file}') from err

        sample = {
            'events': events,
            'file_name': name,
        }

        if self.use_cache:
            try:
                pickle.dump(sample, open(cache_file, 'wb'))
            except Exception as err:
                print('Unable to cache file:', str(err))
    
    return sample
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIDI Dataset loader')
    parser.add_argument('--file_path', type=str, required=True,
                      help='Path to directory containing MIDI files')

    parser.add_argument('--limit', type=int, default=None, 
                      help='Limit the number of files to load')
    
    parser.add_argument('--tokenization', type=str, default='remi',
                      choices=['remi', 'octuple'],
                      help='Tokenization method to use (remi or octuple)')
    
    args = parser.parse_args()
    
    files = glob(f'{args.file_path}/**/*.mid', recursive=True)
    print(f"Found {len(files)} MIDI files")

    if args.limit:
        files = files[:args.limit]
        print(f"Limiting to {args.limit} files")
    
    if len(files) == 0:
        print("Error: No MIDI files found in the specified directory.")
        print("Please make sure the directory exists and contains .mid files.")
        exit(1)
    
    dm = MidiDataModule(files, max_len=2048, tokenization=args.tokenization, skip_unknown_genres=True)
    dm.setup()
    
    print(f"\nDataset splits:")
    print(f"Train files: {len(dm.train_ds.files)}")
    print(f"Valid files: {len(dm.valid_ds.files)}")
    print(f"Test files: {len(dm.test_ds.files)}")
    
    print(f"\nProcessed examples:")
    print(f"Train examples: {len(dm.train_ds)}")
    print(f"Valid examples: {len(dm.valid_ds)}")
    print(f"Test examples: {len(dm.test_ds)}")
    
    dl = dm.train_dataloader()
    
    for batch in dl:
        print(f"\nBatch shape: {batch['input_ids'].shape}")
        print(f"Batch keys: {batch.keys()}")
        print(batch['input_ids'])
        break
