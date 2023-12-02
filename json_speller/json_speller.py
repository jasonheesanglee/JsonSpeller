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
    def __init__(self, list_text, word_dict):
        self.list_text = list_text
        self.word_dict = set(word_dict)
        self.num_word, self.spell_misspells = self.funct(list_text)

    def to_json(self, spell_misspells):
        '''
        ::keyword : misspelled word
        ::bag_of_word : list of possible corrections
        
        This function takes keyword and bag_of_words as an input.
        Exports the input in json format file.
        '''
        if os.path.isfile('misspell_dict.json'):
            with open('misspell_dict.json', 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}
        
        for key, value in spell_misspells.items():
            if key in existing_data:
                # Extend the list in existing_data with new items not already present
                existing_data[key].extend(item for item in value if item not in existing_data[key])
            else:
                existing_data[key] = value
    
        with open('misspell_dict.json', 'w') as file:
            # Dump the existing_data dictionary which now contains the updated data
            json.dump(existing_data, file, indent=4)
    
    def funct(self, list_text):
        '''
        ::list_text : texts(str) in a list
        
        This function takes a list of texts as input.
        
        Returns 
        1. count of each word
        2. dictionary of misspelled_word : [list of possible correction]
        '''
        tokenized_texts = [self.tokenize(text) for text in list_text]
        all_words_flat = [word for text in tokenized_texts for word in text]

        num_word = defaultdict(int) # count how many correctly spelled words in all text
        for word in all_words_flat:
            if word in self.word_dict:
                num_word[word] += 1
        # print(num_word)
        spell_misspells = defaultdict(list) # misspelled_word : [possible corrections]
        for keyword in tqdm(num_word.keys()):
            if keyword == 'wlel':
                print('it is keyword')
            # print(keyword)
            misspellings = self.find_misspellings(all_words_flat, keyword)
            # print(f'\n{misspellings}')
            if misspellings != 'PASS':
                # print(f'{keyword} misspelling_no pass')
                for misspell in misspellings.keys():
                    spell_misspells[misspell].append(keyword)
            
        spell_misspells = dict(spell_misspells)
        self.to_json(spell_misspells)
        return num_word, dict(spell_misspells)
        
    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text)

    def normalize(self, text):
        pattern = r'(.)\1{2,}'
        text = re.sub(pattern, r'\1', text)
        pattern = r'[^a-zA-Z0-9+]'
        return re.sub(pattern, '', text)
    
        
    def find_misspellings(self, all_words, keyword, threshold=2, max_distance=2):
        '''
        this function does not replace the misspells
        '''
        word_count = Counter(all_words)
        bag_of_words = defaultdict(int) # count
        
        for word, count in word_count.items():
            # Normalize the word for comparison
            normalized_word = self.normalize(word)
            
            # Exclude the keyword itself or its normalized version
            if normalized_word.lower() == keyword.lower():
                continue
    
            # Exclude non-alphabetic, all-uppercase, and capitalized words (like proper nouns)
            if not word.isalpha() or word.isupper() or word[0].isupper():
                continue
    
            # Exclude correctly spelled words and very short words
            if normalized_word.lower() in self.word_dict or len(normalized_word) < 4:
                continue
    
            # Check for similarity and count frequency of potential misspellings
            if distance(normalized_word, keyword) <= max_distance:
                bag_of_words[normalized_word] += 1
        
        return 'PASS' if not bag_of_words else bag_of_words

    
    def bert_predict_masked_word(self, sentence, masked_index):
        inputs = tokenizer.encode(sentence, return_tensors='pt')
        mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
    
        token_logits = model(inputs).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
        return [tokenizer.decode([token]) for token in top_tokens]

    def most_contextually_appropriate(self, word_options, sentence, threshold=0.8):
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
        return best_word if best_sim > threshold else None
    
    def character_match_score(self, word1, word2):
        """
        Calculate a score based on the longest common subsequence (LCS) 
        between two words.
        """
        def lcs_length(x, y):
            """Helper function to compute length of LCS."""
            if not x or not y:
                return 0
            elif x[-1] == y[-1]:
                return 1 + lcs_length(x[:-1], y[:-1])
            else:
                return max(lcs_length(x[:-1], y), lcs_length(x, y[:-1]))
    
        lcs_len = lcs_length(word1, word2)
        score = lcs_len / max(len(word1), len(word2)) # Normalized by the length of the longer word
        return score
    
    def find_most_sim_char(self, misspelled_word, replacements, threshold=0.5):
        """
        Find the best replacement for a misspelled word from a list of replacements.
        """
        best_match = None
        highest_score = -1
    
        for replacement in replacements:
            score = self.character_match_score(misspelled_word, replacement)
            if score > highest_score:
                highest_score = score
                best_match = replacement
    
        return best_match if highest_score >= threshold else None        

    def text_replace(self, list_text):
        fixed_text_list = []
        for text in tqdm(list_text):
            new_text = text.split()
            for idx, word in enumerate(text.split()):
                norm_word = self.normalize(word)
                new_text[idx] = norm_word
                if norm_word not in self.spell_misspells.keys():
                    continue
                    
                if len(self.spell_misspells[word]) == 1:
                    similar_word = self.spell_misspells[word][0]
                    new_text[idx] = similar_word

            
            words = new_text.copy()
            for idx, word in enumerate(words):
                if word in self.spell_misspells.keys():
                    # print(f"Word: {word}, Replacements: {self.spell_misspells[word]}")

                    masked_sentence = words.copy()
                    word = self.normalize(word)
                    masked_sentence[idx] = tokenizer.mask_token
                    masked_sentence = ' '.join(masked_sentence)
                    replacements = self.bert_predict_masked_word(masked_sentence, idx)
                    similar_word = None
                    # print(f'BERT searching for the right replacement for word : {word}...')
                    for replacement in replacements:
                        if replacement in self.spell_misspells[word]:
                            similar_word = replacement
                            # print(f'BERT found word replacement for word, the correct word is "{replacement}"\n')
                            break
                            
                    if not similar_word:
                        # print(f'searching contextually appropriate words for word : {word}...')
                        similar_word = self.most_contextually_appropriate(self.spell_misspells[word], text)
                        # if similar_word:
                            # print(f'found the word with the contextual search method, the correct word is "{similar_word}"\n')
                        # break
                        
                    if not similar_word:
                        # print(f'searching word with most similar characters word : {word}...')
                        similar_word = self.find_most_sim_char(word, self.spell_misspells[word])
                        # if similar_word:
                            # print(f'found the word with most similar characters method, the correct word is "{similar_word}"\n')
                        # break
                    
                    if not similar_word:
                        # print(f'searching word with minimum edit distance word : {word}...')
                        similar_word = min(self.spell_misspells[word], key=lambda x: distance(word, x))
                        # if similar_word:
                            # print(f'found the word with minimum edit distance method, the correct word is "{similar_word}"\n')
                        # break
                    
                    # if not similar_word:
                        # print(f"oops, didn't find similar word for {word}")
                        
                    new_text[idx] = similar_word if similar_word else word
        
                    
                    # similar_word = self.most_contextually_appropriate(self.spell_misspells[word], text)
                
            fixed_text_list.append(' '.join(new_text))
                
            
        return fixed_text_list
