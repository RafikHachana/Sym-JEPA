import pretty_midi
from collections import Counter
from torch import Tensor

from src.constants import (
  DEFAULT_VELOCITY_BINS,
  DEFAULT_DURATION_BINS,
  DEFAULT_TEMPO_BINS,
  DEFAULT_POS_PER_QUARTER,
  CHORD_KEY
)


from src.constants import (
  MAX_BAR_LENGTH,
  MAX_N_BARS,

  PAD_TOKEN,
  UNK_TOKEN,
  BOS_TOKEN,
  EOS_TOKEN,
  MASK_TOKEN,

  TIME_SIGNATURE_KEY,
  BAR_KEY,
  POSITION_KEY,
  INSTRUMENT_KEY,
  PITCH_KEY,
  VELOCITY_KEY,
  DURATION_KEY,
  TEMPO_KEY)



class Tokens:
  def get_instrument_tokens(key=INSTRUMENT_KEY):
    tokens = [f'{key}_{pretty_midi.program_to_instrument_name(i)}' for i in range(128)]
    tokens.append(f'{key}_drum')
    return tokens

  def get_chord_tokens(key=CHORD_KEY, qualities = ['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'None']):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    chords = [f'{root}:{quality}' for root in pitch_classes for quality in qualities]
    chords.append('N:N')

    tokens = [f'{key}_{chord}' for chord in chords]
    return tokens

  def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
    denominators = [2, 4, 8, 16]
    time_sigs = [f'{p}/{q}' for q in denominators for p in range(1, MAX_BAR_LENGTH*q + 1)]
    tokens = [f'{key}_{time_sig}' for time_sig in time_sigs]
    return tokens

  def get_midi_tokens(
    instrument_key=INSTRUMENT_KEY, 
    time_signature_key=TIME_SIGNATURE_KEY,
    pitch_key=PITCH_KEY,
    velocity_key=VELOCITY_KEY,
    duration_key=DURATION_KEY,
    tempo_key=TEMPO_KEY,
    bar_key=BAR_KEY,
    position_key=POSITION_KEY
  ):
    instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

    pitch_tokens = [f'{pitch_key}_{i}' for i in range(128)] + [f'{pitch_key}_drum_{i}' for i in range(128)]
    velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
    duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
    tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
    bar_tokens = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)]
    position_tokens = [f'{position_key}_{i}' for i in range(MAX_BAR_LENGTH*4*DEFAULT_POS_PER_QUARTER)]

    time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)

    return (
      time_sig_tokens +
      tempo_tokens + 
      instrument_tokens + 
      pitch_tokens + 
      velocity_tokens + 
      duration_tokens + 
      bar_tokens + 
      position_tokens
    )

from collections import Counter

class _Vocab:
    def __init__(self, counter, specials=("<unk>",), min_freq=1):
        # Sort tokens by frequency (descending) and then alphabetically
        sorted_tokens = sorted(
            [tok for tok, freq in counter.items() if freq >= min_freq],
            key=lambda x: (-counter[x], x)
        )

        # Prepend special tokens
        self.itos = list(dict.fromkeys(list(specials) + sorted_tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        # Define default index (usually unk)
        self.default_index = self.stoi.get("<unk>", None)

    def set_default_index(self, idx):
        self.default_index = idx

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx):
        return self.itos[idx]

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def lookup_tokens(self, indices):
        return [self.lookup_token(i) for i in indices]

class Vocab:
  def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN], unk_token=UNK_TOKEN):
    self.specials = specials
    
    # Create vocab with special tokens first
    self.vocab = _Vocab(
        counter, 
        special_first=True,
        specials=self.specials
    )

    # Set unknown token index if specified
    if unk_token in specials:
        unk_index = self.vocab[unk_token]  # Using __getitem__ instead of get_stoi
        self.vocab.set_default_index(unk_index)

  def to_i(self, token):
    return self.vocab[token]  # Using __getitem__ instead of get_stoi

  def to_s(self, idx):
    if idx >= len(self.vocab):
      return UNK_TOKEN
    else:
      return self.vocab.lookup_token(idx)  # Using lookup_token instead of get_itos

  def __len__(self):
    return len(self.vocab)

  def encode(self, seq):
    return [self.vocab[token] for token in seq]  # Manual encoding since vocab(seq) might not be supported

  def decode(self, seq):
    if isinstance(seq, Tensor):
      seq = seq.numpy()
    return [self.vocab.lookup_token(idx) for idx in seq]  # Using lookup_token for each index


class RemiVocab(Vocab):
  def __init__(self):
    midi_tokens = Tokens.get_midi_tokens()
    chord_tokens = Tokens.get_chord_tokens()

    self.tokens = midi_tokens + chord_tokens

    counter = Counter(self.tokens)
    super().__init__(counter)
