# MIT License

# Copyright (c) 2023 Jason Heesang Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
.. module:: json_speller
   :synopsis: Module for spelling correction utilizing distilbert, Levenshtine distance.
"""


import os
import re
import json
from tqdm import tqdm
import spacy
from collections import Counter
from Levenshtein import distance
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
nlp = spacy.load('en_core_web_lg')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')

freq_dict = './data/symspell_freq_dict 2.txt'
with open(freq_dict, 'r') as f:
    freq_d = f.read()

freq_d = freq_d.replace('\ufeff', '')
sym_words = {}
for st in freq_d.split('\n'):
    sym_words[st.split()[0]] = int(st.split()[1])

class SpellChecker:
    def __init__(self, list_text, freq_dict):
        self.list_text = list_text
        self.eng_dict = freq_dict

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text)

    def normalize(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def find_misspellings(self, texts, target_word, threshold=2, max_distance=2):
        all_words = [word for text in texts for word in self.tokenize(text)]
        word_count = Counter(all_words)
        bag_of_words = {}
        for word, count in word_count.items():
            # if ' '.join([r'\u{:04X}'.format(ord(ele)) for ele in word]) == ' '.join([r'\u{:04X}'.format(ord(ele)) for ele in target_word]):
            if word == target_word:
                continue
            elif word.lower() == target_word.lower():
                continue
            elif word.lower() in self.eng_dict.keys():
                continue
            elif len(word) == 1:
                continue
            elif word != target_word and count < threshold:
                word = self.normalize(word)
                if distance(word, target_word) <= max_distance:
                    try:
                        bag_of_words[word] += 1
                    except KeyError:
                        bag_of_words[word] = 1

        if bag_of_words == {}:
            return 'PASS'
        else:
            return bag_of_words

    def funct(self, list_text):
        num_word = {}
        for text in list_text:
            for word in text.split():
                if word in self.eng_dict.keys():
                    num_w = text.count(word)
                    try:
                        num_word[word] += num_w
                    except KeyError:
                        num_word[word] = num_w

        spell_misspells = {}
        for keyword in num_word.keys():
            misspellings = self.find_misspellings(list_text, keyword)
            if misspellings == 'PASS':
                continue

            else:
                for misspells in list(misspellings.keys()):
                    try:
                        spell_misspells[misspells].append(keyword)
                    except KeyError:
                        spell_misspells[misspells] = [keyword]

        return num_word, spell_misspells

    def most_contextually_appropriate(self, word_options, sentence):
        doc = nlp(sentence)
        best_word = None
        best_sim = 0
        for option in word_options:
            option_token = nlp(option)[0]
            for token in doc:
                sim = token.similarity(option_token)
                if sim > best_sim:
                    best_sim = sim
                    best_word = option
        return best_word

    def predict_masked_word(self, sentence, masked_index):
        inputs = tokenizer.encode(sentence, return_tensors='pt')
        mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]

        token_logits = model(inputs).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        return [tokenizer.decode([token]) for token in top_tokens]


    def to_json(self, list_text):
        _, misspell_lists = self.funct(list_text)
        if os.path.isfile('./data/misspell_dict.json'):
            with open('./data/misspell_dict.json', 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        for key, value in misspell_lists.items():
            if key in existing_data.keys():
                existing_data[key].extend([i for i in value if i not in existing_data[key]])
            else:
                existing_data[key] = value

        with open('./data/misspell_dict.json', 'w') as file:
            json.dump(misspell_lists, file, indent=4)

    def text_replace(self, list_text):
        _, spell_misspells = self.funct(list_text)

        fixed_text_list = []
        for text in tqdm(list_text):
            new_text = text.split()
            for idx, word in enumerate(text.split()):
                new_text[idx] = self.normalize(word)
                word = self.normalize(word)
                if word not in spell_misspells.keys():
                    continue

                if len(spell_misspells[word]) == 1:
                    similar_word = spell_misspells[word][0]
                    new_text[idx] = similar_word

            words = new_text.copy()
            for idx, word in enumerate(words):
                if word in spell_misspells.keys():
                    masked_sentence = words.copy()
                    masked_sentence[idx] = tokenizer.mask_token
                    masked_sentence = ' '.join(masked_sentence)
                    replacements = self.predict_masked_word(masked_sentence, idx)
                    similar_word = None
                    for replacement in replacements:
                        if replacement in spell_misspells[word]:
                            similar_word = replacement
                            break
                    if not similar_word:
                        for replacement in spell_misspells[word]:
                            if word[0] == replacement[0]:
                                similar_word = replacement
                                break
                    if not similar_word:
                        # similar_word = self.most_contextually_appropriate(spell_misspells[word], text)
                        similar_word = min(spell_misspells[word], key=lambda x: distance(word, x))
                    new_text[idx] = similar_word if similar_word else word

                    # similar_word = self.most_contextually_appropriate(spell_misspells[word], text)

            fixed_text_list.append(' '.join(new_text))
        self.to_json(list_text=list_text)
        return fixed_text_list
