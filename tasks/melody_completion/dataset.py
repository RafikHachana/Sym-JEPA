from torch.utils.data import Dataset
from dataset import MidiDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.utils import shuffle
from uuid import uuid4


class MelodyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, file_paths, batch_size=32, num_workers=4):
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.num_workers = num_workers



    def setup(self, stage=None):
        n_train = int(len(self.file_paths) * 0.9)
        n_val = len(self.file_paths) - n_train
        self.train_dataset = MelodyPredictionDataset(self.file_paths[:n_train], negative_pairs_per_positive=1)
        self.val_dataset = MelodyPredictionDataset(self.file_paths[n_train:], negative_pairs_per_positive=49)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=SeqCollator())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=SeqCollator())

class SeqCollator:
    def __init__(self, pad_token=0):
        self.pad_token = pad_token
        

    def __call__(self, features):
        batch = {}

        all_input = sum([feature['input'] for feature in features], [])
        all_input_padded = pad_sequence([torch.tensor(x) for x in all_input], batch_first=True, padding_value=self.pad_token)

        batch['input'] = all_input_padded
        batch['match'] = torch.tensor(sum([feature['match'] for feature in features], []))
        batch['uuid'] = sum([feature['uuid'] for feature in features], [])

        return batch



class MelodyPredictionDataset(Dataset):
    def __init__(self, file_paths, negative_pairs_per_positive=1):
        self.file_paths = file_paths
        self.negative_pairs_per_positive = negative_pairs_per_positive

        self.midi_dataset = MidiDataset(file_paths, tokenization='octuple', max_len=2048)

        self.bos, self.eos = self.midi_dataset.get_bos_eos_events()

        # Convert from tensor to list
        self.bos = self.bos.tolist()
        self.eos = self.eos.tolist()



    def __len__(self):
        return len(self.midi_dataset)

    def __getitem__(self, idx):
        return self._get_pairs(idx)

    def _get_pairs(self, idx):
        instance = self.midi_dataset[idx]

        result = {
            "input": [],
            "match": [],
            "uuid": []
        }
        first_half = instance['input_ids'][:len(instance['input_ids'])//2].tolist()
        second_half = instance['input_ids'][len(instance['input_ids'])//2:].tolist()

        first_half_uuid = uuid4()

        # print("First half: ", first_half)
        # print("Second half: ", second_half)
        # print("BOS: ", self.bos)
        # print("EOS: ", self.eos)

        # Get the positive pair
        result['input'].append(self.bos + first_half + self.eos + second_half + self.eos)
        result['match'].append(1)
        result['uuid'].append(first_half_uuid)


        for i in range(self.negative_pairs_per_positive):
            # Get a random index
            random_idx = np.random.randint(0, len(self.midi_dataset))
            while random_idx == idx:
                random_idx = np.random.randint(0, len(self.midi_dataset))

            # Get the negative pair
            negative_match = self.midi_dataset[random_idx]['input_ids'].tolist()
            result['input'].append(self.bos + first_half + self.eos + negative_match[len(negative_match)//2:] + self.eos)
            result['match'].append(0)
            result['uuid'].append(first_half_uuid)

        # Shuffle the result
        result['input'], result['match'] = shuffle(result['input'], result['match'])
        return result


        
