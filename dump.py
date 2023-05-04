import pickle
from tqdm import tqdm
import json
import random

# with open("Hindi_PL_Bert/hindi_phonemes_word_vocab_updated.txt", encoding="utf-8") as f:
#     hindi_phoneme_vocab = f.read().splitlines()

# hindi_phoneme_vocab_dict = {i:w for i,w in enumerate(hindi_phoneme_vocab)}
# hindi_phoneme_vocab_dict[len(hindi_phoneme_vocab)] = hindi_phoneme_vocab_dict[3039]
# hindi_phoneme_vocab_dict[3039] = "<formula>"

# dump = {i+1: {'word': v, 'token': k} for i, (k,v) in enumerate(hindi_phoneme_vocab_dict.items())}
# dump[3039], dump[3040] = dump[3040], dump[3039]

# print(dump[3039])

# with open("token_maps_hindi.pkl", "wb") as f:
    # pickle.dump(dump, f)

# del hindi_phoneme_vocab

with open("token_maps_hindi.pkl", "rb") as f:
    h = pickle.load(f)

hindi_phoneme_vocab_dict = {i['word']: i['token'] for i in h.values()}

del h

with open("Hindi_PL_Bert/hindi_dump_phonemes_updated.csv", encoding="utf") as f:
    for sentence in tqdm(f):
        sentence = sentence.strip()
        p = sentence.split()
        ids = [hindi_phoneme_vocab_dict[i] for i in p]
        phonemized = {'input_ids': ids, 'phonemes': p}
        
        with open("dataset.jsonl", "a") as output_file:
            output_file.write(json.dumps(phonemized, ensure_ascii=False))
            output_file.write("\n")
