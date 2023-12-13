# JsonSpeller

## Training in progress

### Under Development

# Main Functions


1. `spell_check(list_text, deep_search=True, context_search=True, sim_char_search=True)`<br>
    Replaces the misspelled words in the given list of texts using various strategies without training.
    
    - `list_text` : python list of texts.
    - `deep_search` : default=True | enables / disables deep-learning search.
    - `context_search` : default=True | enables / disables context search.
    - `sim_char_search` : default=True | enables / disables similar character search.

2. `train_check(list_text, deep_search=True, context_search=True, sim_char_search=True)`<br>
    Train the words in the given list and updates the misspell.json
    Then replaces the misspelled words in the given list of texts using various strategies.
    
    - `list_text` : python list of texts.
    - `deep_search` : default=True | enables / disables deep-learning search.
    - `context_search` : default=True | enables / disables context search.
    - `sim_char_search` : default=True | enables / disables similar character search.

3. `train(list_text)`<br>
    Train the words in the given list and updates the misspell.json
    - `list_text` : python list of texts.


# Material used in Training

- [COVID19 Tweets](https://www.kaggle.com/datasets/gpreda/covid19-tweets) by [@Gabirel Preda](https://github.com/gabrielpreda)
- [Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets) by [@Viktor Stepanenko](https://www.kaggle.com/vstepanenko)
- [Bitcoin tweets](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) by [@Alexandre Bouillet](https://github.com/alexandrebouillet)
- More in line...


# Comparison
(to be observed once training is done.)

1. JsonSpeller

2. autocorrect

3. textblob

4. symspellpy

5. pyspellcheck



# Why did I make this?


### 답답하면 니가 뛰던지
의 '니'가 되었습니다.

마음에 쏙드는 한국어 맞춤법교정기가 없어서... 만들어보는 중이고... 시험삼아 영어를 먼저 만들어 보았습니다!<br><br>


While working on my final project, I realized that there are not much great spellcheckers for Korean.<br>
Therefore, I have decided to create one.<br>
Named it JsonSpeller as it works based on JSON file. <s>Not because I am Jason</s><br>

Before diving into my spellchecker, here are the results of different spellcheckers.<br>
*We had around 2,000 data to spellcheck.*

# Other models
|Model|Time|Result|Tested By|
|---|---|---|---|
|T5|2.5 hours|It does a good job, but some hate-speech are labeled as "hate-speech" and some as it is|@JasonHeesangLee|
|T5 - detailed prompt|4 hours|I'll have to modify the prompt again to try, but for now, it shows only **####Hello####** x million times...|@JasonHeesangLee|
|ChatGPT API - simple prompt|2.5 hours|It is good, but too good... For hate-speech, it cleanses the term and explains the situation. (May damage the original if used as is)<br>Most importantly, it costs...|@SionBang|
|ChatGPT - detailed prompt|2.5 hours|It is good, hate-speech labeled as **[OFF]**, when a single word is all there is to the text -> Explains that word. (May damage the original if used as is)<br>Also, it costs...|@JasonHeesangLee|
|Rule-based Spacing Module with [khaiii POS-tagger](https://github.com/kakao/khaiii) by [Kakao Corp](https://www.kakaocorp.com/page/)|22 seconds|95% spacing correctly done (Maybe able to reach upto 100% if the **rule** can be even more developed). However, it doesn't correct the spellings, but only reconcatenate the consonants and vowels if they are separated (It happens often in typing fast). It may be useful if ensembled with other spell checking models.|@JasonHeesangLee|
|[symspellpy-ko](https://github.com/HeegyuKim/symspellpy-ko)|20 minutes|A model developed over [symspellpy](https://github.com/mammothb/symspellpy).<br>As the model checks the spell based on the external dictionary, some new terms could be leftout or wrongly corrected. (If a perfect wordlist exists... I believe this would the best model in theory, but we all know that the perfect wordlist doesn't exist.)|@JasonHeesangLee|
|[Pusan-University Spellchecking API](http://speller.cs.pusan.ac.kr/)|40 ~ 60 minutes|Performs well at spacing, sanitizing profanity, and separating sentences / Doesn't understand neologisms, people's names, etc. and returns incorrect spellings. Also, it [cannot be used for deep-learning](https://www.yna.co.kr/view/AKR20230707041700051) as the service is free, and is offered for public.|@NayoungBae|
|[soynlp](https://github.com/lovit/soynlp)|20 minutes for training<br>(30,000 sentences), 3 seconds for inference|Some neologisms were well maintained while some were not. For other normal terms, it also returns bad spacing.|@SionBang|
|[hanspell](https://github.com/ssut/py-hanspell)|No Idea|Error on all teammates' environment.|ALL|


***symspellpy-ko*** would have been the best model among the models above, only if the perfect word-list, including neologisms.<br><br>

Then I thought "***What if we make typos for words on purpose, and put them in the dictionary to replace them?***" <br>
But creating typos would be too inefficient, even if generate them with LLMs.<br>Below is the modified version of this ***Thought*** I had.<br>(The current version I have developed can only perform spell check on English, and it will be developed for Korean soon.)<br>Please kindly go through them and tell me if this method would be too inefficient or if there is any problems I haven't thought of.<br>*Disclaimer We are working on extracting the keywords from the users’ daily records and segment them into “Positive” and “Negative” emotions, as explained in this [discussion](https://www.kaggle.com/discussions/general/456900) and this **JsonSpeller** was initially developed solely for this project.*<br><br>

# Facts & Hypothesis
A keyword term must appear at least twice to be considered a keyword.<br>
For nouns that appear only once, if the spelling is correct (if it is in the Korean word-list), we will extract it as a keyword, otherwise we will exclude it.<br>
Even if it is included, the user may say: "*What is this? It's not a word I wrote down.*"<br><br>

# How to
1. Prepare tweet texts or any texts that are written informally (not the news articles) for pretraining.<br>
2. List-up the words that appears more than once.<br>The ratio could be adjusted after passing certain number of pretraining - when we have enough data to set threshold.<br>
3. Extract the words in the ***possible-misspelled-terms category*** of those correct words from the text and put them into a dictionary in the form of key(misspelled-words) - value_array(possible-correct-terms).<br><t>***3-1.*** What is *possible-misspelled-terms category* and *possible-correct-terms category*?
    - ***possible-misspelled-terms category*** : misspelled terms like "wolrd", "word", "owrld", etc. when the correct word is "world".<br>However, the words in the this category must appear only once in the text.<br>If it appears more than once, the word is added to the list mentioned above.
    - Since there are real words like *word*, it is likely that the user meant world when "world" appears multiple times and "word" only appears for twice.
    - We exclude the actual words that are in the dictionary because we don't want future uses of ***word*** to be recognized as ***world***.<br>
</t><t>
For example the **key - value_array** will look like : **"wword" : ["word", "world", "work", ... etc.]**</t> <br>
4. Iterate over the dictionary value_array words and replace them with nouns from the string.
  
    
# Pros
- If this is applied in real-world service, it is possible to create separate JSON files of frequently used typos for each user and use it continuously. (Personalization of misspell processing?)
- Since the process only runs for words recognized as nouns in the sentence (Korean Language Only), the processing speed does not increase proportionally as the sentence gets longer.
- Many variants of typos can be quickly collected when combining these personalized JSON files.
- No need to build a dictionary of neologisms (because the words people use often will be in known words - appears more than 2 times)<br>
- We can use different dataset to pretrain on a certain topic - like data science, sports, medical, ... etc. whatever topic you think of.

# Cons
- Some words could be leftout or misjudged.<br>
- It's likely to take a long time to go through each word.
- But... will there really be that many typo variants for one word...?
- ***유저들의 행동이 모든게 계획대로 돌아가야 함.***


Last but not least, a great ***THANK YOU*** my teammates for being patient with me!
[@nayoungbae](https://www.kaggle.com/nayoungbae) [@chanhyukhan](https://www.kaggle.com/chanhyukhan) [@bangsioni](https://www.kaggle.com/bangsioni)
