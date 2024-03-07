import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
import gensim
import numpy as np




def load_model_and_data(path_to_model, path_to_input_words, language):
    """
    Load a Word2Vec model and input words from macht.sprache.

    Args:
        path_to_model (str): Path to the Word2Vec model file.
        path_to_input_words (str): Path to the input words JSON file.
        language (str): Language for selecting input words.

    Returns:
        gensim.models.Word2Vec: Loaded Word2Vec model.
        pd.Series: Input words filtered by the specified language.
    """
    w2v = gensim.models.Word2Vec.load(path_to_model).wv

    input_words_en_de = pd.read_json(path_to_input_words)
    input_words = input_words_en_de[input_words_en_de['lemma_lang'] == language]['lemma'].reset_index(drop=True)
    return w2v, input_words


def generate_similar_words(w2v, input_words, nr_similar_words, similarity_threshold):
    """
    Generate lists of similar words to macht.sprache words.

    Args:
        w2v (gensim.models.Word2Vec): Word2Vec model.
        input_words (pd.Series): Input words.
        nr_similar_words (int): Number of similar words to retrieve.
        similarity_threshold (float): Minimum similarity threshold.

    Returns:
        pd.DataFrame: DataFrame with input words, similar words, and similarity values.
    """
    input_and_similar_words = pd.DataFrame(input_words.items(), columns=['index','input_word'])
    input_and_similar_words['similar_words'] = ''
    input_and_similar_words['words with similarity value'] = ''


    for index, item in input_words.items():
        try: 
            most_similar_words = w2v.most_similar(item, topn=nr_similar_words)
            input_and_similar_words.at[index, 'similar_words'] = [tuple[0] for tuple in most_similar_words if tuple[1] > similarity_threshold]
            input_and_similar_words.at[index, 'words with similarity value'] =   [tuple for tuple in most_similar_words]
        except: 
            input_and_similar_words.at[index, 'similar_words'] = np.nan
            

    input_and_similar_words.dropna(inplace=True) # remove all the rows of input words that could not be found in the lexicon
    input_and_similar_words.drop(labels=['index'], axis=1, inplace=True)
    input_and_similar_words[['input_word', 'words with similarity value']].to_csv("similar_words_with_similarity_value", index=False)
    return input_and_similar_words



def filter_for_sensitivity(w2v, input_and_similar_words, buzzwords):
    """
    Filter similar words for sensitivity based on the similarity to social justice buzzwords.
    Sort the words according to their sensitivity score.

    Args:
        w2v (gensim.models.Word2Vec): Word2Vec model.
        input_and_similar_words (pd.DataFrame): DataFrame with input words and similar words.
        buzzwords (list): List of social justice buzzwords.

    Returns:
        pd.DataFrame: DataFrame with similar words, sensitivity scores, and input words.
    """ 
    sensitive_words_df = pd.DataFrame(columns=['similar_word', 'sensitivity_score', 'input_word'])

    for index, row in input_and_similar_words.iterrows():
        similar_words = row['similar_words']
        for similar_word in similar_words: 
            sensitive_similarity = 0
            for buzzword in buzzwords: 
                sensitive_similarity = sensitive_similarity + w2v.similarity(similar_word, buzzword)
            # Weighting the sensitive_similarity
            weighted_sensitive_similarity = sensitive_similarity/len(buzzwords)
            sensitive_words_df.loc[len(sensitive_words_df)] = [similar_word, round(weighted_sensitive_similarity, 3), row['input_word']]
            


    # Make sure the newly found terms do not occur more than once in the output
    sensitive_words_df = sensitive_words_df.groupby(['similar_word', 'sensitivity_score']).agg({'input_word': ', '.join}).reset_index()
    # Sort the terms according to their sensitivity score
    sensitive_words_df.sort_values(by=['sensitivity_score'], ascending=False, inplace=True)

    return sensitive_words_df



def sensitive_buzzwords_approach(nr_similar_words=50, similarity_threshold=0.6,
    language='en', buzzwords=['discrimination', 'power', 'political'], 
    path_to_model= os.path.join('models', 'word2vec_test.model'),
    path_to_input_words=os.path.join('macht.sprache_input', 'macht.sprache_words.json')):
    

    # Load the pretrained model and the terms from macht.sprache
    w2v, input_words = load_model_and_data(path_to_model, path_to_input_words, language)
    # Generate a dataframe of similar words to the words from macht.sprache
    input_and_similar_words = generate_similar_words(w2v, input_words, nr_similar_words, similarity_threshold)
    # Filter similar words for sensitivity based on the similarity to social justice buzzwords. Sort the words according to their sensitivity score.
    sensitive_words_df = filter_for_sensitivity(w2v, input_and_similar_words, buzzwords)
    # Output the list of new terms (with their sensitivity score)
    sensitive_words_df.to_csv("output_buzzwords_approach.csv", index=False)
    return sensitive_words_df


if __name__ == "__main__":
    sensitive_buzzwords_approach()