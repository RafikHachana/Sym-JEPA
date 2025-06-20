import os
import pandas as pd
import unicodedata
from torch.utils.data import Dataset
from dataset import MidiDataset


from torch.utils.data import Dataset
from dataset import MidiDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from pprint import pprint


DIFFICULTY_IN_META = True  # Set to False if difficulty is not in metadata
class PerformerClassificationDataModule(pl.LightningDataModule):
    def __init__(self, midi_base_path, metadata_path, batch_size=32, num_workers=4, top_k_performers=20, top_k_composers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        assert not (top_k_composers and top_k_performers), "Cannot select the top performers AND top composers at the same time"

        self.metadata = pd.read_csv(metadata_path)
        print(self.metadata['composer'].value_counts()[:top_k_composers].keys())
        if top_k_composers:
            top_composers = self.metadata['composer'].value_counts().nlargest(top_k_composers).index
            self.metadata = self.metadata[self.metadata['composer'].isin(top_composers)]

        if top_k_performers:
            top_performers = self.metadata['artist'].value_counts().nlargest(top_k_performers).index
            self.metadata = self.metadata[self.metadata['artist'].isin(top_performers)]
        self.metadata_path = metadata_path
        self.midi_base_path = midi_base_path

        self.full_length = len(self.metadata)

        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)

        self.setup_done = False

    def setup(self, stage=None):
        if self.setup_done:
            return
        
        n_train = int(self.full_length * 0.9)
        n_val = self.full_length - n_train
        self.train_dataset = ATEPPDataset(self.midi_base_path, self.metadata, start_index=0, end_index=n_train)
        self.val_dataset = ATEPPDataset(self.midi_base_path, self.metadata, start_index=n_train, end_index=self.full_length)

        self.setup_done = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=SeqCollator())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=SeqCollator())

class SeqCollator:
    def __init__(self, pad_token=0):
        self.pad_token = pad_token
        

    def __call__(self, features):
        batch = {}

        all_input = [feature['input_ids'] for feature in features]
        all_input_padded = pad_sequence(all_input, batch_first=True, padding_value=self.pad_token)

        batch['input_ids'] = all_input_padded
        batch['performer'] = torch.tensor([feature['performer'] for feature in features])
        batch['composer'] = torch.tensor([feature['composer'] for feature in features])

        if DIFFICULTY_IN_META:
            batch['difficulty'] = torch.tensor([feature['difficulty'] for feature in features])

        return batch

class ATEPPDataset(Dataset):
    def __init__(self, midi_base_path, metadata, start_index=0, end_index=None, load_midi=True):
        # print(f"Loading ATEPP dataset from {metadata_path} and MIDI files from {midi_base_path}")
        self.metadata = metadata
        self.midi_base_path = midi_base_path

        self.start_index = start_index
        self.end_index = end_index if end_index is not None else len(self.metadata)

        self.metadata = self.metadata.iloc[start_index:end_index].reset_index(drop=True)


        self.file_paths = []

        performer_count = {}
        composer_count = {}
        for index, row in self.metadata.iterrows():
            self.file_paths.append(os.path.join(self.midi_base_path, unicodedata.normalize('NFD', row['midi_path'])))
            if row['artist'] not in performer_count:
                performer_count[row['artist']] = 0
            if row['composer'] not in composer_count:
                composer_count[row['composer']] = 0
            
            composer_count[row['composer']] += 1
            performer_count[row['artist']] += 1

        if load_midi:
            self.midi_dataset = MidiDataset(
                midi_files=self.file_paths,
                max_len=512,
                # batch_size=32,
                # num_workers=4,
                # pin_memory=True,
            )
        else:
            self.midi_dataset = None

        self.unique_performers = self.metadata['artist'].unique()
        self.performer_to_idx = {performer: idx for idx, performer in enumerate(self.unique_performers)}
        self.unique_composers = self.metadata['composer'].unique()
        self.composer_to_idx = {composer: idx for idx, composer in enumerate(self.unique_composers)}

        pprint(performer_count)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]


        # Normalize Unicode paths to handle encoding differences between CSV and filesystem
        normalized_midi_path = unicodedata.normalize('NFD', row['midi_path'])
        normalized_base_path = unicodedata.normalize('NFD', self.midi_base_path)
        
        midi_path = os.path.join(normalized_base_path, normalized_midi_path)
        
        assert os.path.exists(midi_path) and os.path.isfile(midi_path), f"MIDI file not found at {midi_path}"

        difficulty = row.get('difficulty', None)
        if difficulty is not None:
            difficulty = int(difficulty)

        return {
            "performer": self.performer_to_idx[row['artist']],
            "composer": self.composer_to_idx[row['composer']],
            "midi_path": midi_path,
            "difficulty": difficulty,
            **self.midi_dataset[idx]
        }
