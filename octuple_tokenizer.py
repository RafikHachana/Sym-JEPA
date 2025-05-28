from tokenizer_base import TokenizerBase
from constants import *
import numpy as np
import pretty_midi
import traceback
import pdb
from typing import List

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import sys
import io
import zipfile
import miditoolkit
import random
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager
from vocab import Vocab
from collections import Counter

pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = False
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 100000  # window length max
sample_overlap_rate = 1
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

data_zip = None
output_file = None


lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
manager = Manager()
midi_dict = manager.dict()


# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]


ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]


def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator



def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]

            print("Onset", time_to_pos(note.start), "End", time_to_pos(note.end))
            # The structure is here
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding




def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash


def F(file_name):
    try_times = 10
    midi_file = None
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        # traceback.print_exc()
        # print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        # print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    try:
        e = MIDI_to_encoding(midi_obj)
        if len(e) == 0:
            # print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None
        if ts_filter:
            allowed_ts = t2e(time_signature_reduce(4, 4))
            if not all(i[6] == allowed_ts for i in e):
                print('ERROR(TSFILT): ' + file_name + '\n', end='')
                return None
        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            lock_set.release()
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                      file_name + ' == ' + dup_file_name + '\n', end='')
                return None
        output_str_list = []
        sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
        for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
            L = max(p, 0)
            R = min(p + sample_len_max, len(e)) - 1
            bar_index_list = [e[i][0]
                              for i in range(L, R + 1) if e[i][0] is not None]
            bar_index_min = 0
            bar_index_max = 0
            if len(bar_index_list) > 0:
                bar_index_min = min(bar_index_list)
                bar_index_max = max(bar_index_list)
            offset_lower_bound = -bar_index_min
            offset_upper_bound = bar_max - 1 - bar_index_max
            # to make bar index distribute in [0, bar_max)
            bar_index_offset = random.randint(
                offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
            e_segment = []
            for i in e[L: R + 1]:
                if i[0] is None or i[0] + bar_index_offset < bar_max:
                    e_segment.append(i)
                else:
                    break
            tokens_per_note = 8
            output_words = ([('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)])
            output_str_list.append(' '.join(output_words))


        # no empty
        if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
       
        # print('SUCCESS: ' + file_name + '\n', end='')
        return output_str_list[0].split()
    except BaseException as e:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False

def get_standard_vocab_tokens():
    tokens = []
    for i in range(bar_max):
        tokens.append(f"<0-{i}>")
    for i in range(beat_note_factor * max_notes_per_bar * pos_resolution):
        tokens.append(f"<1-{i}>")
    for i in range(max_inst + 1 + 1):
        tokens.append(f"<2-{i}>")
    for i in range(2 * max_pitch + 1 + 1):
        tokens.append(f"<3-{i}>")
    for i in range(duration_max * pos_resolution):
        tokens.append(f"<4-{i}>")
    for i in range(v2e(max_velocity) + 1):
        tokens.append(f"<5-{i}>")
    for i in range(len(ts_list)):
        tokens.append(f"<6-{i}>")
    for i in range(b2e(max_tempo) + 1):
        tokens.append(f"<7-{i}>")

    return tokens

def get_max_vector():
    return [
        bar_max,
        beat_note_factor * max_notes_per_bar * pos_resolution,
        max_inst + 2,
        2 * max_pitch + 2,
        duration_max * pos_resolution,
        v2e(max_velocity) + 1,
        len(ts_list),
        b2e(max_tempo) + 1,
    ]



def get_special_tokens():
    return ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]

def token_to_value(token):
    if token in get_special_tokens():
        return -1
    try:
        return int(token.split('-')[1][:-1])
    except:
        print(token)
        raise ValueError(f"Unknown token: {token}")



class OctupleVocab(Vocab):
    def __init__(self):
        self.tokens = get_standard_vocab_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter, specials=get_special_tokens(), unk_token="<unk>")
        
        

class OctupleTokenizer(TokenizerBase):
    def version():
        return 'octuple_v1'
    
    def __init__(self, file, strict=False):
        super().__init__(file, skip_pm=True)
        
        assert isinstance(file, str), "File must be a string"
        self.file_path = file



    def get_events(self):
        result = F(self.file_path)
        if not result:
            raise ValueError("Could not parse MIDI file")
        return result
    
    @staticmethod
    def get_vocab():
        return OctupleVocab()
    
    @staticmethod
    def get_bos_eos_tokens():
        return ["<s>", "</s>"]
    
    @staticmethod
    def get_mask_token():
        return "<mask>"
    
    @staticmethod
    def get_unk_token():
        return "<unk>"
    
    @staticmethod
    def get_pad_token():
        return "<pad>"

    @staticmethod
    def wrap_for_jepa(tokens: str):
        bos, eos = OctupleTokenizer.get_bos_eos_tokens()
        return f"{bos*8} {tokens} {eos*8}"
    
    



def breakout_octuple(tokens: List[str]):
    assert len(tokens) % 8 == 0, "Tokens must be a multiple of 8"
    result = {
        "bar": [],
        "pitch": [],
        "duration": [],
        "velocity": [],
        "time_signature_ind": [],
        "tempo": [],
        "onset": [],
        "instrument": [],
        "time_signature_denom": [],
        "time_signature_numer": [],
    }

    for i in range(len(tokens) // 8):
        first_token = tokens[i * 8]

        if first_token in get_special_tokens():
            result["bar"].append(None)
            result["pitch"].append(None)
            result["duration"].append(None)
            result["velocity"].append(None)
            result["time_signature_ind"].append(None)
            result["tempo"].append(None)
            result["onset"].append(None)
            result["instrument"].append(None)
            result["time_signature_denom"].append(None)
            result["time_signature_numer"].append(None)
            continue
        

        for token in tokens[i * 8: (i + 1) * 8]:
            if token.startswith("<0-"):
                result["bar"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<1-"):
                result["onset"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<2-"):
                result["instrument"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<3-"):
                result["pitch"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<4-"):
                result["duration"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<5-"):
                result["velocity"].append(int(token.split("-")[1][:-1]))
            elif token.startswith("<6-"):
                result["time_signature_ind"].append(int(token.split("-")[1][:-1]))
                result["time_signature_denom"].append(ts_list[int(token.split("-")[1][:-1])][1])
                result["time_signature_numer"].append(ts_list[int(token.split("-")[1][:-1])][0])
            elif token.startswith("<7-"):
                result["tempo"].append(int(token.split("-")[1][:-1]))
            else:
                raise ValueError(f"Unknown token: {token}")
        
    
    
    return result

# Extras

def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj


if __name__ == "__main__":
    path = 'dataset/clean_midi/AC DC/Highway To Hell.mid'
    tokenizer = OctupleTokenizer(path)
    events = MIDI_to_encoding(miditoolkit.midi.parser.MidiFile(path))
    print(events)
    pdb.set_trace()




# TODO: Add a sample script to test the tokenizer

if __name__ == "__main__":
    tokenizer = OctupleTokenizer("dataset/lmd_full/0/07eb0e9e50e969cf7dc4019125bd59ac.mid")
    events = tokenizer.get_events()
    print(events)
    breakout = breakout_octuple(events)
    print({k: v[:10] for k, v in breakout.items()})
