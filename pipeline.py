import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
import gensim
import numpy as np

# Paths
path_to_model =  os.path.join('models', 'word2vec_test.model')
path_to_input_words = os.path.join('macht.sprache_input', 'macht.sprache_words.json')

# Load model & macht.sprache words
w2v = gensim.models.Word2Vec.load("word2vec_test.model").wv

input_words_en_de = pd.read_json(path_to_input_words)
input_words_en = input_words_en_de[input_words_en_de['lemma_lang'] == 'en']['lemma'].reset_index(drop=True)
input_words_de = input_words_en_de[input_words_en_de['lemma_lang'] == 'de']['lemma'].reset_index(drop=True)



# Generate lists of similar words to macht.sprache words
nr_similar_words = 20 

input_and_similar_words_en = pd.DataFrame(input_words_en.items(), columns=['index','input_word'])
input_and_similar_words_en['similar_words'] = ''


for index, item in input_words_en.items():
    try: 
        most_similar_words = w2v.most_similar(item, topn=nr_similar_words)
        #print(item, most_similar_words)
        input_and_similar_words_en.at[index, 'similar_words'] = [tuple[0] for tuple in most_similar_words]  
    except: 
        input_and_similar_words_en.at[index, 'similar_words'] = np.nan
        # TODO: store words that are not in the lexicon to output them in the end?

input_and_similar_words_en.dropna(inplace=True) # remove all the rows of input words that could not be found in the lexicon
input_and_similar_words_en.drop(labels=['index'], axis=1, inplace=True)
print(input_and_similar_words_en)



# Filter the lists for sensitive terms  
# 1. Approach: calculate the similarity to social justice buzzwords and rank the newly found words accordingly
buzzwords = ['discrimination', 'power', 'political']
sensitive_words_df = pd.DataFrame(columns=['similar_word', 'sensitive_similarity', 'input_word'])



for index, row in input_and_similar_words_en.iterrows():
    similar_words = row['similar_words']
    for similar_word in similar_words: 
        sensitive_similarity = 0
        for buzzword in buzzwords: 
            sensitive_similarity = sensitive_similarity + w2v.similarity(similar_word, buzzword)
            # Weighting the sensitive_similarity, TODO: better weighting?
            weighted_sensitive_similarity = sensitive_similarity/len(buzzwords)
            sensitive_words_df.loc[len(sensitive_words_df)] = [similar_word, round(weighted_sensitive_similarity, 3), row['input_word']]




# 2. Approach: ask an LLM to assign a sensitivity score
            


# Make sure the newly found terms do not occur more than once in the output
sensitive_words_df.drop_duplicates(subset=['similar_word'], inplace=True)
sensitive_words_df.sort_values(by=['sensitive_similarity'], ascending=False, inplace=True)


# Output the list of new terms (with their sensitivity score)
print(sensitive_words_df)
sensitive_words_df.to_csv("similar_sensitive_words.csv", index=False)
