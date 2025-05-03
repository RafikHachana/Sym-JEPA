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
    
    # @abstractmethod
    # def events_to_midi(self, events):
    #     """Convert events back to MIDI"""
    #     pass 