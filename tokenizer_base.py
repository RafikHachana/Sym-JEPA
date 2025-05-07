from abc import ABC, abstractmethod
import pretty_midi

class TokenizerBase(ABC):
    @abstractmethod
    def __init__(self, file, **kwargs):
        if isinstance(file, pretty_midi.PrettyMIDI):
            self.pm = file
        else:
            self.pm = pretty_midi.PrettyMIDI(file)
    
    @abstractmethod
    def get_events(self):
        """Return list of events representing the MIDI file"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_vocab():
        """Return vocabulary for tokenization"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_bos_eos_tokens():
        """Return beginning and end of sequence tokens"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_mask_token():
        """Return mask token"""
        pass

    @staticmethod
    @abstractmethod
    def get_unk_token():
        """Return unknown token"""
        pass

    @staticmethod
    @abstractmethod
    def get_pad_token():
        """Return pad token"""
        pass