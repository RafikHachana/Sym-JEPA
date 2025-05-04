import json

genre_mapping_path = '../../metadata/midi_genre_map.json'

with open(genre_mapping_path, 'r') as f:
  genre_mapping = json.load(f)

print(len(genre_mapping['topmagd']))

genres = []
n_genres = []
for key, value in genre_mapping['topmagd'].items():
  genres += value
  n_genres.append(len(set(value)))

  if len(set(value)) > 3:
    print(set(value))
print(len(genres))

length_freq = {}
for genre in n_genres:
  if genre in length_freq:
    length_freq[genre] += 1
  else:
    length_freq[genre] = 1

print(length_freq)

print("Unique genres: ", len(set(genres)))

print(set(genres))