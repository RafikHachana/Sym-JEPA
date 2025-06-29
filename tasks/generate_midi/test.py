from .model import MusicDecoder
from dataset import MidiDataset
from glob import glob
import os
import torch

def test():
    model = MusicDecoder()
    model.load_state_dict(torch.load("/root/Sym-JEPA/symjepa-decode/yjtb0fxg/checkpoints/last.ckpt")["state_dict"])
    model.eval()
    model.to("cuda")

    ## Load the dataset
    file_paths = glob(os.path.join("/root/Sym-JEPA/dataset/clean_midi", "**/*.mid"), recursive=True)[:5]
    midi_dataset = MidiDataset(file_paths, tokenization='octuple', max_len=2048)


    # Get the first entry
    entry = midi_dataset[0]
    input_ids = entry['input_ids']
    # target_mask = entry['target_mask']
    # target_ids = entry['target']
    # uuid = entry['uuid']

    # Generate the instrument
    result = model.generate_instrument(input_ids, 0)

    # Generate the instrument


if __name__ == "__main__":
    test()