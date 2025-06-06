from octuple_tokenizer import OctupleVocab
import torch

def random_masking(input_ids, octuple_breakout, mask_prob=0.2):
    """
    Masks random tokens in the input sequence.
    """
    context_mask = torch.bernoulli(torch.ones(input_ids.size()) * mask_prob)
    target_mask = ~context_mask

    context_mask = context_mask.to(input_ids.device)
    target_mask = target_mask.to(input_ids.device)

    return context_mask, target_mask

def contiguous_masking(input_ids, octuple_breakout, context_ratio=0.5):
    """
    Masks contiguous tokens in the input sequence.
    """
    # Random contiguous masking
    seq_length = input_ids.size()
    mask_step = 8
    mask_start_max = int(seq_length // mask_step * context_ratio) * mask_step - mask_step
    assert mask_start_max % mask_step == 0, f"mask_start_max {mask_start_max} is not divisible by mask_step {mask_step}"
    mask_start = torch.randint(mask_step, mask_start_max, (1,)) // mask_step * mask_step
    mask_size = int(seq_length // mask_step * (1 - context_ratio)) * mask_step
    

    mask = torch.zeros(seq_length, dtype=torch.bool)
    mask[mask_start:mask_start+mask_size] = True

    target_mask_start = torch.randint(1, (mask_start + mask_size) // mask_step, (1,)) * mask_step
    target_mask_size = torch.randint(1, (mask_start + mask_size - target_mask_start) // mask_step, (1,)) * mask_step
    target_mask = torch.zeros(seq_length, dtype=torch.bool)
    target_mask[target_mask_start:target_mask_start+target_mask_size] = True

    return mask, target_mask

def instrument_masking(input_ids, octuple_breakout):
    unique_instruments = set([x for x in octuple_breakout["instrument"] if x is not None])

    if len(unique_instruments) == 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)

    # Pick a random instrument
    instrument = unique_instruments[torch.randint(0, len(unique_instruments), (1,))]

    context_mask = torch.zeros_like(input_ids)

    for i, x in enumerate(octuple_breakout['instrument']):
        if x is None:
            continue
        if x == instrument:
            context_mask[i*8:i*8+8] = True
    
    target_mask = ~context_mask

    return context_mask, target_mask


def transpose_masking(input_ids, octuple_breakout):
    semitones = torch.randint(-5, 6, (1,))
    
    octuple_breakout['pitch'] = [None if x is None else x + semitones if x <= 127 else None for x in octuple_breakout['pitch']]

    octuple_vocab = OctupleVocab()

    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue

        encoded = octuple_vocab.encode(f"<3-{x}>")
        input_ids[i*8+3] = encoded[0]

    return input_ids

def rhythmic_noise_masking(input_ids, octuple_breakout):

    octuple_breakout['duration'] = [x + torch.randint(-2, 1, (1,)) if x is not None else None for x in octuple_breakout['duration']]

    octuple_vocab = OctupleVocab()

    for i, x in enumerate(octuple_breakout['duration']):
        if x is None:
            continue
        if x < 0:
            x = 0
        if x > 127:
            x = 127

        encoded = octuple_vocab.encode(f"<4-{x}>")
        input_ids[i*8+4] = encoded[0]

    return input_ids

def mask_pitch_classes(input_ids, octuple_breakout):
    pitch_classes = set([x % 12 for x in octuple_breakout['pitch'] if x is not None and x <= 127])

    if len(pitch_classes) == 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)

    # Pick a random pitch class
    pitch_class = pitch_classes[torch.randint(0, len(pitch_classes), (1,))]

    context_mask = torch.zeros_like(input_ids)

    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue
        if x % 12 == pitch_class:
            context_mask[i*8:i*8+8] = True

    target_mask = ~context_mask

    return context_mask, target_mask

def mask_octaves(input_ids, octuple_breakout):
    octaves = set([x // 12 for x in octuple_breakout['pitch'] if x is not None and x <= 127])

    if len(octaves) == 1:
        # Fallback to random masking
        return random_masking(input_ids, octuple_breakout)
    
    # Pick a random octave
    octave = octaves[torch.randint(0, len(octaves), (1,))]

    context_mask = torch.zeros_like(input_ids)

    for i, x in enumerate(octuple_breakout['pitch']):
        if x is None or x > 127:
            continue
        if x // 12 == octave:
            context_mask[i*8:i*8+8] = True

    target_mask = ~context_mask

    return context_mask, target_mask



    
    
    
    

