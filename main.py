import os
from sensitive_buzzwords import sensitive_buzzwords_approach
from sensitive_cloud import main 

# Parameters
nr_similar_words = 50
similarity_threshold = 0.6
language = 'en' # or 'de'
buzzwords = ['discrimination', 'political']

# Paths
path_to_model =  os.path.join('models', 'word2vec_test.model')
path_to_input_words = os.path.join('macht.sprache_input', 'macht.sprache_words.json')


if __name__ == "__main__":
    sensitive_words_df = sensitive_buzzwords_approach(nr_similar_words, similarity_threshold, language, buzzwords, path_to_model, path_to_input_words)
    print(sensitive_words_df)
    #main()