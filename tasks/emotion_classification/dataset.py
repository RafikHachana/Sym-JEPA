import os
import pandas as pd
import unicodedata
from torch.utils.data import Dataset
from src.dataset import MidiDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class EMOPIADataModule(pl.LightningDataModule):
    def __init__(self, midi_paths, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.midi_paths = midi_paths

        self.setup_done = False

        self.full_length = len(midi_paths)

    def setup(self, stage=None):
        if self.setup_done:
            return
        
        n_train = int(self.full_length * 0.995)
        n_val = self.full_length - n_train
        self.train_dataset = EMOPIADataset(self.midi_paths[:n_train])
        self.val_dataset = EMOPIADataset(self.midi_paths[n_train:])

        self.setup_done = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=SeqCollator())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=max(1, self.batch_size//50), num_workers=self.num_workers, collate_fn=SeqCollator())

class SeqCollator:
    def __init__(self, pad_token=0):
        self.pad_token = pad_token
        

    def __call__(self, features):
        batch = {}

        all_input = [feature['input_ids'] for feature in features]
        all_input_padded = pad_sequence(all_input, batch_first=True, padding_value=self.pad_token)

        batch['input_ids'] = all_input_padded
        batch['emotional_quadrant'] = torch.tensor([feature['emotional_quadrant'] for feature in features])
        batch['arousal_value'] = torch.tensor([feature['arousal_value'] for feature in features])
        batch['valence_value'] = torch.tensor([feature['valence_value'] for feature in features])

        return batch
    
def emotional_quadrant_to_idx(quadrant):
    """Convert emotional quadrant to index."""
    quadrant_map = {
        'Q1': 0,
        'Q2': 1,
        'Q3': 2,
        'Q4': 3
    }
    return quadrant_map[quadrant]

class EMOPIADataset(Dataset):
    def __init__(self, midi_paths):
        self.file_paths = midi_paths

        self.midi_dataset = MidiDataset(
            midi_files=self.file_paths,
            max_len=512,
            # batch_size=32,
            # num_workers=4,
            # pin_memory=True,
        )

        self.emotional_quadrant = [emotional_quadrant_to_idx(os.path.basename(path).split('_')[0]) for path in midi_paths]

        self.arousal_value = [1 if os.path.basename(path).split('_')[1] in ['Q2', 'Q4'] else 0 for path in midi_paths]
        self.valence_value = [1 if os.path.basename(path).split('_')[1] in ['Q1', 'Q1'] else 0 for path in midi_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return {
            "emotional_quadrant": self.emotional_quadrant[idx],
            "arousal_value": self.arousal_value[idx],
            "valence_value": self.valence_value[idx],
            **self.midi_dataset[idx]
        }
