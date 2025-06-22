import random
from typing import Dict, Tuple
from octuple_tokenizer import OctupleVocab
import torch


transpose_range = 12
instrument_range = 128

# TODO: Add drums to instrument range

def _get_transpose_id(semitones):
    return (semitones + transpose_range) % transpose_range +1

def _get_instrument_id(instrument):
    return instrument + transpose_range + 1

def random_masking(input_ids, octuple_breakout, mask_prob=0.1):
    """
    Masks random tokens in the input sequence.
    """
    context_mask = torch.bernoulli(torch.ones(input_ids.size()) * mask_prob).bool()
    target_mask = ~context_mask

    context_mask = context_mask.to(input_ids.device)
    target_mask = target_mask.to(input_ids.device)

    return context_mask, target_mask, input_ids, 0

def contiguous_masking(input_ids, octuple_breakout, context_ratio=0.5):
    """
    Masks contiguous tokens in the input sequence.
    """
    # Random contiguous masking
    seq_length = input_ids.size()[0]
    mask_step = 8
    mask_start_max = int(seq_length // mask_step * context_ratio) * mask_step - mask_step
    assert mask_start_max % mask_step == 0, f"mask_start_max {mask_start_max} is not divisible by mask_step {mask_step}"
    mask_start = torch.randint(mask_step, mask_start_max, (1,)) // mask_step * mask_step
    mask_size = int(seq_length // mask_step * (1 - context_ratio)) * mask_step
    

    mask = torch.zeros(seq_length, dtype=torch.bool)
    mask[mask_start:mask_start+mask_size] = True

    target_mask_start = torch.randint(mask_start.item() // mask_step, (mask_start.item() + mask_size) // mask_step, (1,)) * mask_step
    try:
        target_mask_size = torch.randint(1, (mask_start.item() + mask_size - target_mask_start.item()) // mask_step, (1,)) * mask_step
    except:
        target_mask_size = 8
    target_mask = torch.ones(seq_length, dtype=torch.bool)
    target_mask[target_mask_start:target_mask_start+target_mask_size] = False

    return mask, target_mask, input_ids, 0

def instrument_masking(input_ids, octuple_breakout):
    unique_instruments = list(set([x for x in octuple_breakout["instrument"] if x is not None and x < instrument_range]))

    if len(unique_instruments) <= 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)

    # Pick a random instrument
    instrument = unique_instruments[torch.randint(0, len(unique_instruments), (1,))]

    context_mask = torch.zeros_like(input_ids).bool()
    target_mask = torch.ones_like(input_ids).bool()

    n_masked = 0
    for i, x in enumerate(octuple_breakout['instrument']):
        if x is None:
            continue
        if x == instrument:
            context_mask[i*8:i*8+8] = True
            if n_masked < 100:
                target_mask[i*8:i*8+8] = False
            n_masked += 1
    

    return context_mask, target_mask, input_ids, _get_instrument_id(instrument)


def transpose_masking(input_ids, octuple_breakout):
    semitones = torch.randint(-5, 7, (1,))
    
    octuple_breakout['pitch'] = [None if x is None else x + semitones if x <= 127 else None for x in octuple_breakout['pitch']]

    octuple_vocab = OctupleVocab()

    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue

        encoded = octuple_vocab.encode(f"<3-{x}>")
        input_ids[i*8+3] = encoded[0]
    mask = torch.zeros_like(input_ids).bool()
    return mask, mask, input_ids, _get_transpose_id(semitones.item())

def rhythmic_noise_masking(input_ids, octuple_breakout):

    octuple_breakout['duration'] = [x + torch.randint(-3, 2, (1,)) if x is not None else None for x in octuple_breakout['duration']]

    octuple_vocab = OctupleVocab()

    for i, x in enumerate(octuple_breakout['duration']):
        if x is None:
            continue
        if x < 0:
            x = 0
        if x > 7:
            x = 7

        encoded = octuple_vocab.encode(f"<4-{x}>")
        input_ids[i*8+4] = encoded[0]
    mask = torch.zeros_like(input_ids).bool()
    return mask, mask, input_ids, 0

def no_masking(input_ids, octuple_breakout):    
    return torch.zeros_like(input_ids).bool(), torch.zeros_like(input_ids).bool(), input_ids, 0

def mask_pitch_classes(input_ids, octuple_breakout):
    pitch_classes = list(set([x % 12 for x in octuple_breakout['pitch'] if x is not None and x <= 127]))

    if len(pitch_classes) <= 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)

    # Pick a random pitch class
    pitch_class = pitch_classes[torch.randint(0, len(pitch_classes), (1,))]

    context_mask = torch.zeros_like(input_ids).bool()
    target_mask = torch.ones_like(input_ids).bool()

    n_masked = 0
    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue
        if x % 12 == pitch_class:
            context_mask[i*8:i*8+8] = True
            if n_masked < 100:
                target_mask[i*8:i*8+8] = False
            n_masked += 1

    return context_mask, target_mask, input_ids, 0

def mask_octaves(input_ids, octuple_breakout):
    octaves = list(set([x // 12 for x in octuple_breakout['pitch'] if x is not None and x <= 127]))

    if len(octaves) <= 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)
    
    # Pick a random octave
    octave = octaves[torch.randint(0, len(octaves), (1,))]

    context_mask = torch.zeros_like(input_ids).bool()
    target_mask = torch.ones_like(input_ids).bool()

    n_masked = 0

    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue
        if x // 12 == octave:
            context_mask[i*8:i*8+8] = True
            if n_masked < 100:
                target_mask[i*8:i*8+8] = False
            n_masked += 1

    return context_mask, target_mask, input_ids, 0


class RandomMaskGenerator:
    def __init__(self, probs: Dict[str, float]):
        self.probs = probs

        assert abs(sum(self.probs.values()) - 1.0) < 1e-8, f"Probabilities must sum to 1.0, got {sum(self.probs.values())}"

        self.mask_functions = {
            "random": random_masking,
            "contiguous": contiguous_masking,
            "instrument": instrument_masking,
            "transpose": transpose_masking,
            "rhythmic_noise": rhythmic_noise_masking,
            "pitch_classes": mask_pitch_classes,
            "octaves": mask_octaves,
            "none": no_masking,
        }

        assert all(mask_name in self.mask_functions for mask_name in self.probs), f"All mask names must be in {self.mask_functions}"

    def __call__(self, input_ids, octuple_breakout, random_mask_prob=0.1, contiguous_context_ratio=0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        mask_function = random.choices(list(self.probs.keys()), weights=list(self.probs.values()))[0]
        if mask_function == "random":
            result = self.mask_functions[mask_function](input_ids, octuple_breakout, mask_prob=random_mask_prob)
        elif mask_function == "contiguous":
            result = self.mask_functions[mask_function](input_ids, octuple_breakout, context_ratio=contiguous_context_ratio)
        else:
            result = self.mask_functions[mask_function](input_ids, octuple_breakout)
        
        return *result, mask_function

if __name__ == "__main__":
    # mask_generator = RandomMaskGenerator({"random": 0.1, "contiguous": 0.3, "instrument": 0.1, "transpose": 0.1, "rhythmic_noise": 0.2, "pitch_classes": 0.1, "octaves": 0.1})
    mask_generator = RandomMaskGenerator({"transpose": 1.0})
    input_ids = torch.randint(0, 128, (64,))
    octuple_breakout = {"instrument": [1, 2, 1, 2, 1, 1, 1, 1], "pitch": [60, 75, 60, 70, 60, 58, 60, 61], "duration": [1, 2, 1, 5, 1, 1, 1, 1]}
    print(mask_generator(input_ids, octuple_breakout))


    


    
    
    
    

