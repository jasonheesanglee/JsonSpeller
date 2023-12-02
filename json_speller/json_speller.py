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
   :synopsis: Module for spelling correction utilizing distilbert, Levenshtine distance, character similarity comparison, editdistance.
"""


import os
import re
import json
import pandas as pd
from tqdm import tqdm
import spacy
from collections import Counter, defaultdict
from Levenshtein import distance
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
nlp = spacy.load('en_core_web_lg')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')


word_dictionary = pd.read_json('./data/word_dict.json', typ='series')
word_dict = word_dictionary.keys()



freq_d = freq_d.replace('\ufeff', '')
sym_words = {}
for st in freq_d.split('\n'):
    sym_words[st.split()[0]] = int(st.split()[1])

class SpellChecker:
    """
    A class that provides spell checking and correction functionality.

    Attributes:
        list_text (list): A list of input texts (sentences) to be processed.
        word_dict (set): A set containing correctly spelled words for reference.
        num_word (dict): A dictionary that holds the count of correctly spelled words.
        spell_misspells (dict): A dictionary mapping misspelled words to their potential corrections.
    """
    
    def __init__(self, list_text, word_dict):
        """
        Initializes the SpellChecker with a list of texts and a reference word dictionary.
        """
        self.list_text = list_text
        self.word_dict = set(word_dict)
        self.num_word, self.spell_misspells = self.funct(list_text)

    def to_json(self, spell_misspells):
        """
        Exports the mappings of misspelled words to their potential corrections to a JSON file.

        Args:
            spell_misspells (dict): A dictionary containing misspelled words and their corrections.
        """
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
        """
        Processes the list of texts to count correctly spelled words and find misspellings.

        Args:
            list_text (list): A list of texts to be processed.

        Returns:
            tuple: A tuple containing two dictionaries - one for word counts and one for misspellings.
        """
        tokenized_texts = [self.tokenize(text) for text in list_text]
        all_words_flat = [word for text in tokenized_texts for word in text]
        word_count = Counter(all_words_flat)
        num_word = defaultdict(int)  
        for word, count in word_count.items():
            if count > 2:
                self.word_dict.add(word)

            if word in self.word_dict:
                num_word[word] += count

        # Process for misspellings
        spell_misspells = defaultdict(list)
        for keyword in num_word:
            # Skip known words
            if word in self.word_dict:
                continue
            misspellings = self.find_misspellings(all_words_flat, keyword)
            if misspellings != 'PASS':
                for misspell in misspellings.keys():
                    spell_misspells[misspell].append(keyword)
            
        spell_misspells = dict(spell_misspells)
        self.to_json(spell_misspells)
        return num_word, dict(spell_misspells)
        
    def tokenize(self, text):
        """
        Tokenizes the given text into words.

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of words extracted from the text.
        """
        return re.findall(r'\b\w+\b', text)

    def normalize(self, text):
        """
        Normalizes the given text by reducing repeated characters and removing non-alphanumeric characters.

        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text.
        """
        pattern = r'(.)\1{2,}'
        text = re.sub(pattern, r'\1', text)
        pattern = r'[^a-zA-Z0-9+]'
        return re.sub(pattern, '', text)
    
        
    def find_misspellings(self, all_words, keyword, threshold=2, max_distance=2):
        """
        Identifies misspelled words in a list of words based on their similarity to a given keyword.

        Args:
            all_words (list): A list of words to be checked.
            keyword (str): The word to compare against for finding misspellings.
            threshold (int): The minimum count for a word to be considered.
            max_distance (int): The maximum allowed Levenshtein distance for a word to be a potential misspelling.

        Returns:
            dict or str: A dictionary of potential misspellings or 'PASS' if none are found.
        """
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
        """
        Predicts possible replacements for a masked word in a sentence using BERT.

        Args:
            sentence (str): The text containing the masked word.
            masked_index (int): The index of the masked word in the sentence.

        Returns:
            list: A list of predicted words by BERT for the masked position.
        """

        inputs = tokenizer.encode(sentence, return_tensors='pt')
        mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
    
        token_logits = model(inputs).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
        return [tokenizer.decode([token]) for token in top_tokens]

    def most_contextually_appropriate(self, word_options, sentence, threshold=0.8):
        """
        Finds the most contextually appropriate word from a list of options for a given sentence.

        Args:
            word_options (list): A list of word options for replacement.
            sentence (str): The sentence in which the word is to be replaced.
            threshold (float): The minimum similarity score for a word to be considered.

        Returns:
            str or None: The most appropriate word if found, otherwise None.
        """

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
        Calculates a similarity score based on the longest common subsequence between two words.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: The calculated similarity score.
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
        Finds the best replacement for a misspelled word based on character similarity.

        Args:
            misspelled_word (str): The misspelled word.
            replacements (list): A list of possible replacements.
            threshold (float): The minimum similarity score for a word to be considered.

        Returns:
            str or None: The best replacement word if found, otherwise None.
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
        """
        Replaces misspelled words in the given list of texts using various strategies.

        Args:
            list_text (list): A list of texts to process.

        Returns:
            list: A list of texts with misspelled words replaced.
        """
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
                    print(f"Word: {word}, Detected possible replacements: {self.spell_misspells[word]}") # uncommented to show the result

                    masked_sentence = words.copy()
                    word = self.normalize(word)
                    masked_sentence[idx] = tokenizer.mask_token
                    masked_sentence = ' '.join(masked_sentence)
                    replacements = self.bert_predict_masked_word(masked_sentence, idx)
                    similar_word = None
                    print(f'BERT searching for the right replacement for word : {word}...') # uncommented to show the result
                    for replacement in replacements:
                        if replacement in self.spell_misspells[word]:
                            similar_word = replacement
                            print(f'BERT found word replacement for word, the correct word is "{replacement}"\n') # uncommented to show the result
                            break
                            
                    if not similar_word:
                        print(f'searching contextually appropriate words for word : {word}...') # uncommented to show the result
                        similar_word = self.most_contextually_appropriate(self.spell_misspells[word], text)
                        if similar_word: # uncommented to show the result
                            print(f'found the word with the contextual search method, the correct word is "{similar_word}"\n') # uncommented to show the result
                        
                    if not similar_word:
                        print(f'searching word with most similar characters word : {word}...') # uncommented to show the result
                        similar_word = self.find_most_sim_char(word, self.spell_misspells[word])
                        if similar_word: # uncommented to show the result
                            print(f'found the word with most similar characters method, the correct word is "{similar_word}"\n') # uncommented to show the result
                    
                    if not similar_word:
                        print(f'searching word with minimum edit distance word : {word}...') # uncommented to show the result
                        similar_word = min(self.spell_misspells[word], key=lambda x: distance(word, x))
                        if similar_word: # uncommented to show the result
                            print(f'found the word with minimum edit distance method, the correct word is "{similar_word}"\n') # uncommented to show the result
                    
                    if not similar_word: # uncommented to show the result
                        print(f"oops, didn't find similar word for {word}") # uncommented to show the result
                        
                    new_text[idx] = similar_word if similar_word else word
                
            fixed_text_list.append(' '.join(new_text))
                
            
        return fixed_text_list
        
    


# example_texts = [
#     "Hello this is the new world",
#     "this rowld is baeutiful",
#     "Adam and Eve came to this new world",
#     "thiis wword is beautiful",
#     'hello hi',
#     'hellohi',
#     'what are you sayinggg',
#     'what is your purpose?',
#     "don't tell me what to do",
#     'what do you want from me',
#     'This friday should be fun',
#     "this doesn't work at all?",
#     'yaaaaaay today is friiiiidaaaaaayyyyyyyy',
#     'yo whta are you doing',
#     'why is this not working so well',
#     'I believe this should wokr welwl'
# ]


# spell_checker = SpellChecker(list_text=example_texts, word_dict=word_dict)
# final = spell_checker.text_replace(example_texts)
# print(example_texts[:2])
# print()
# for original, fixed in zip(example_texts, final):
#     print(f'{original} - original')
#     print(f'{fixed} - fixed\n')
