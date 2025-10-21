import json
import hashlib
import glob
import os
from tqdm import tqdm
def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
    

with open('metadata/midi_genre_map.json', 'r') as f:
    genre_map = json.load(f)


midi_glob = 'dataset/lmd_full/**/*.mid'


print(next(iter(genre_map['masd'].keys())))

# f9eb546a3ca09f543d0c7bc6d809602b
# exit()

found_topmagd = 0
found_magd = 0
found_masd = 0


count = 0

unique_md5s = set()

print(len(glob.glob(midi_glob, recursive=True)))
for midi_path in tqdm(glob.glob(midi_glob, recursive=True)):
    md5 = get_md5(midi_path)
    file_id = midi_path.split("/")[-1].split(".")[0]
    md5 = file_id
    if md5 == 'f9eb546a3ca09f543d0c7bc6d809602b':
        print("FOUND IT")
    # if md5 in unique_md5s:
    #     continue
    unique_md5s.add(md5)
    if md5 in genre_map['topmagd']:
        found_topmagd += 1
    if md5 in genre_map['magd']:
        found_magd += 1
    if md5 in genre_map['masd']:
        found_masd += 1

print(f'Found {found_topmagd} out of {len(genre_map["topmagd"])} topmagd')
print(f'Found {found_magd} out of {len(genre_map["magd"])} magd')
print(f'Found {found_masd} out of {len(genre_map["masd"])} masd')

print(f'Found {len(unique_md5s)} unique md5s')






