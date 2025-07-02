from .model import MusicDecoder
from dataset import MidiDataset
from glob import glob
import os

import torch
import miditoolkit
from octuple_tokenizer import encoding_to_MIDI, OctupleVocab, token_to_value, MIDI_to_encoding

def test():
    model = MusicDecoder()
    model.load_state_dict(torch.load("/root/Sym-JEPA/symjepa-decode/yyu9kmz9/checkpoints/last.ckpt")["state_dict"])
    model.eval()
    model.to("cuda")

    ## Load the dataset
    file_paths = glob(os.path.join("/root/Sym-JEPA/dataset/clean_midi", "**/*.mid"), recursive=True)[:5]
    midi_dataset = MidiDataset(file_paths, tokenization='octuple', max_len=4096)


    # Get the first entry
    entry = midi_dataset[2]
    input_ids = entry['input_ids']
    print("File path: ", entry['file_path'])
    # return
    # target_mask = entry['target_mask']
    # target_ids = entry['target']
    # uuid = entry['uuid']

    # Generate the instrument
    octuple_midi_encoded = model.generate_instrument(input_ids, 26)

    # octuple_midi_encoded = input_ids

    octuple_midi_decoded = OctupleVocab().decode(octuple_midi_encoded.squeeze(0).cpu().numpy().tolist())

    print("Length of encoding: ", len(octuple_midi_decoded))

    encoding_flattened = [token_to_value(token) for token in octuple_midi_decoded]


    print("Length of encoding: ", len(encoding_flattened))

    print(octuple_midi_decoded)

    encoding_flattened = sum([encoding_flattened[i:i+8] for i in range(0, len(encoding_flattened), 8) if all(x != -1 for x in encoding_flattened[i:i+8])], [])

    print("Length of encoding after cleaning: ", len(encoding_flattened))

    encoding = [encoding_flattened[i:i+8] for i in range(0, len(encoding_flattened), 8)]
    min_bar = min(x[0] for x in encoding)

    # Remove any bar offset
    for i in range(len(encoding)):
        encoding[i][0] -= min_bar

    print("Length of encoding: ", len(encoding))

    assert len(encoding) * 8 == len(encoding_flattened), f"Encoding length mismatch after grouping tokens: {len(encoding)} * 8 != {len(encoding_flattened)}"

    # Generate the instrument
    midi_obj = encoding_to_MIDI(encoding)
    midi_obj.dump(f"test.mid")


if __name__ == "__main__":
    test()