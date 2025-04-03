from tokenizer_base import TokenizerBase
from constants import *
import numpy as np
import pretty_midi

class OctupleTokenizer(TokenizerBase):
    def version():
        return 'octuple_v1'
    
    def __init__(self, file, strict=False):
        super().__init__(file)
        
        if strict and len(self.pm.time_signature_changes) == 0:
            raise ValueError("Invalid MIDI file: No time signature defined")
            
        self.resolution = self.pm.resolution
        self.note_items = []
        self._read_items()
        self._quantize_items()
        
        if strict and len(self.note_items) == 0:
            raise ValueError("Invalid MIDI file: No notes found, empty file.")
            
    def _read_items(self):
        for instrument in self.pm.instruments:
            for note in instrument.notes:
                self.note_items.append({
                    'measure': 0,  # Will be set in quantize
                    'position': self.pm.time_to_tick(note.start),
                    'program': 128 if instrument.is_drum else instrument.program,
                    'pitch': note.pitch + 128 if instrument.is_drum else note.pitch,
                    'duration': self.pm.time_to_tick(note.end) - self.pm.time_to_tick(note.start),
                    'velocity': note.velocity
                })
                
    def _quantize_items(self):
        # Quantize positions to grid
        ticks_per_pos = self.resolution / OCTUPLE_POS_RESOLUTION
        for item in self.note_items:
            # Quantize position
            pos = item['position']
            quantized_pos = round(pos / ticks_per_pos) * ticks_per_pos
            item['position'] = int(quantized_pos)
            
            # Calculate measure
            time = self.pm.tick_to_time(pos)
            measure = 0
            for i, downbeat in enumerate(self.pm.get_downbeats()):
                if time < downbeat:
                    measure = i
                    break
            item['measure'] = measure
            
            # Quantize duration
            duration = item['duration']
            quantized_duration = max(1, round(duration / ticks_per_pos))
            item['duration'] = int(quantized_duration)
            
    def get_events(self):
        events = []
        for item in sorted(self.note_items, key=lambda x: (x['measure'], x['position'], x['program'], x['pitch'])):
            events.extend([
                f"{BAR_KEY}_{item['measure']}",
                f"{POSITION_KEY}_{item['position']}",
                f"{INSTRUMENT_KEY}_{item['program']}",
                f"{PITCH_KEY}_{item['pitch']}",
                f"{DURATION_KEY}_{item['duration']}",
                f"{VELOCITY_KEY}_{item['velocity']}",
                f"{TIME_SIGNATURE_KEY}_{self._get_time_signature(item['position'])}",
                f"{TEMPO_KEY}_{self._get_tempo(item['position'])}"
            ])
        return events
        
    def _get_time_signature(self, tick):
        time = self.pm.tick_to_time(tick)
        for ts in self.pm.time_signature_changes:
            if time >= ts.time:
                return f"{ts.numerator}/{ts.denominator}"
        return "4/4"  # Default time signature
        
    def _get_tempo(self, tick):
        time = self.pm.tick_to_time(tick)
        tempo = self.pm.get_tempo_changes()
        tempo_at_time = tempo[0][-1]  # Default to last tempo
        for t, bpm in zip(*tempo):
            if time >= t:
                tempo_at_time = bpm
        return int(tempo_at_time) 