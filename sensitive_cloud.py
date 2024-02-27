import gensim.downloader as api
import numpy as np

# Next 3 functions are used to identify and project on "political axes" in the embedding space
def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors. It is independent of the magnitude of the vectors and focuses solely on their direction

    Args:
        v1 (np.ndarray)
        v2 (np.ndarray)

    Returns:
        float: The cosine similarity between the two vectors. Values range from -1 (exactly opposite),
        through 0 (orthogonal), to 1 (exactly the same).
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def create_vec_axis(vec, positive_words, negative_words): 
    """
    Creates a vector axis by subtracting the mean vector of negative words from the mean vector of positive words.
    This function effectively forms a semantic axis in the vector space, representing a specific dimension
    of meaning (e.g. happy vs. sad). The axis is created by averaging the vectors of words
    considered to have positive sentiment or meaning and then subtracting the average vector of words with
    a negative sentiment or meaning.

    Args:
        vec (gensim.models.keyedvectors.KeyedVectors): The word vectors model.
        positive_words (list of str): Words considered to have a positive sentiment or meaning.
        negative_words (list of str): Words considered to have a negative sentiment or meaning.

    Returns:
        np.ndarray: The vector representing the semantic axis defined by the positive and negative words.
    """
    positive_vector = np.mean([vec[word] for word in positive_words if word in vec.key_to_index], axis=0)
    negative_vector = np.mean([vec[word] for word in negative_words if word in vec.key_to_index], axis=0)
    return positive_vector - negative_vector


def project_word_on_vec(embedding_space, word, axis):
    """
    Projects a word vector onto a specified axis, calculating its position relative to that axis.
    This function projects the vector representation of a word onto a predefined semantic axis (e.g., a happy-sad axis)
    to quantify the relevance or association of the word with the semantic dimension represented by the axis.
    This is achieved by calculating the cosine similarity between the word's vector and the axis vector.

    Args:
        vector (gensim.models.keyedvectors.KeyedVectors): The word vectors model.
        word (str): The word to project onto the axis.
        axis (np.ndarray): The vector representing the axis onto which the word vector is projected.

    Returns:
        float: The projection of the word vector onto the axis, indicating the word's association with the axis.
    """
    # get the vector representation of the word according to embedding space
    word_vector = embedding_space[word]
    # calculate cosine similarity between the word and the semantic axis in question
    projection = cosine_similarity(word_vector, axis)
    return projection


def find_best_dataset_dim(datasets, dims, test_words):
    """
    Identify the best dataset-dimension combination for identifying political sensitivity,
    based on the lowest average absolute error in projections compared to expected sensitivity labels.
    
    Args:
        datasets: dict of dataset names to gensim.models.keyedvectors.KeyedVectors objects for evaluation.
        dims: list of dicts, each containing "positive" and "negative" keys with word lists defining political axes for each dimension.
        test_words: dict of test words with labels indicating political sensitivity (1) or neutrality (0).
        
    Returns:
        A tuple containing the name of the best dataset, the name/number of the best dimension, and the lowest average absolute error.
    """
    min_error = np.inf
    best_dataset = None
    best_dim = None

    for dataset_name, dataset in datasets.items():
        for dim_name, dim_values in dims.items():
            pos_words = dim_values["positive"]
            neg_words = dim_values["negative"]
            total_error = 0
            num_words_evaluated = 0
            axis = create_vec_axis(dataset, pos_words, neg_words)
            
            for word, label in test_words.items():
                if word not in dataset:
                    continue  # Skip words not in the dataset
                projection = project_word_on_vec(dataset, word, axis)
                # label 1 for politically sensitive, 0 for neutral
                expected_value = 1 if label == 1 else 0
                # error is calculated by difference of projection value and expected value
                error = np.abs(np.abs(projection) - expected_value)
                total_error += error
                num_words_evaluated += 1

            if num_words_evaluated > 0:
                average_error = total_error / num_words_evaluated
                # we want to return the dataset - dimension combination with the best performance on our test data
                if average_error < min_error:
                    min_error = average_error
                    best_dataset = dataset_name
                    best_dim = dim_name

    return best_dataset, best_dim, min_error

def calculate_political_sensitivity(dataset, dimension, sensitive_word):
    """
    Calculates and ranks the political sensitivity of words similar to a given sensitive word.
    
    Args:
        dataset: gensim.models.keyedvectors.KeyedVectors, the dataset containing word vectors.
        dimension: dict, with "positive" and "negative" keys and lists of words as values defining a political axis.
        sensitive_word: str, the politically sensitive word to analyze.
        
    Returns:
        A list of the top 10 words most similar in political sensitivity to the given word.
    """
    
    # Ensure the sensitive word is in the dataset
    if sensitive_word not in dataset.key_to_index:
        return []
    
    # Create the political axis
    axis = create_vec_axis(dataset, dimension["positive"], dimension["negative"])

    
    # Find the 50 most similar words to the sensitive word
    most_similar_words = dataset.most_similar(sensitive_word, topn=50)
    
    # Project each similar word onto the political axis and calculate the difference from the base word's projection
    word_projections = []
    for word, _ in most_similar_words:
        # calculate projection score (high values indicate political connotation)
        projection = abs(project_word_on_vec(dataset, word, axis))
        word_projections.append((word, projection))
    
    # Order the words by their projection score (descending)
    word_projections.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top 10 words
    return [word for word, _ in word_projections[:10]]

def load_pretrained_embeddings():

    datasets = {
        "Twitter": api.load("glove-twitter-25"),
        "Wikipedia": api.load("glove-wiki-gigaword-200"),
        "Google_news": api.load("word2vec-google-news-300")
    }
    return datasets

def define_political_dimensions():
    # Your defined dimensions here (as in your original pipeline)
    dims = {
       "positive":{"positive":["socialism", "welfare", "equality", "redistribution", "taxes", "healthcare", "universal", "subsidies", "cooperative"], "negative":["capitalism", "deregulation", "privatization", "markets", "taxcuts", "insurance",  "monopoly", "inequity", "exploitation"]},
        "social": {"positive":[ "equality", "rights", "feminism", "queer", "diversity", "reform", "inclusion", "justice", "empowerment", "tolerance"], "negative":["tradition", "patriotism", "nationalism", "family", "heritage", "order", "conservatism", "segregation", "exclusion", "inequality"]},
        "environment": {"positive":["climate", "renewable", "conservation", "sustainable", "green", "ecology","biophilia", "restoration", "permaculture", "biodiversity"], "negative":["coal", "drilling", "deregulation", "growth", "nuclear", "oil","deforestation", "pollution", "extinction", "waste"]},
        "foreign": {"positive":["cooperation", "rights", "globalization", "NATO", "trade", "peace","diplomacy", "multilateralism", "aid", "openness"],"negative":["sovereignty", "borders", "tariffs", "nationalism", "security", "immigration","isolationism", "protectionism", "conflict", "xenophobia"]},
        "governance": {"positive":["democracy", "transparency", "liberty", "rights", "press", "justice","accountability", "participation", "equality", "rule of law"], "negative":["power", "surveillance", "control", "censorship", "state", "security","corruption", "authoritarianism", "inequality", "impunity"]}
    }
    return dims

def load_words():
    words_to_analyze = [
        "ability", "ableism", "aboriginal", "ageism", "agency", "ally", "ancestors", "antisemitism", "asylum", "barbarian"
    ]
    test_words = {
        # Politically Loaded Terms (1)
        "fascism": 1,"socialism": 1,"impeachment": 1,"referendum": 1,"nationalism": 1,"abortion": 1,"censorship": 1,"sanctions": 1,"tariffs": 1,"protest": 1,
    
        # Politically Neutral Terms (0)
        "library": 0,"restaurant": 0,"mountain": 0,"ocean": 0,"piano": 0,"calendar": 0,"umbrella": 0,"refrigerator": 0,"chair": 0,"window": 0
    }
    return words_to_analyze, test_words

def main():
    # Load pretrained word embeddings
    datasets = load_pretrained_embeddings()

    # Define political dimensions
    dims = define_political_dimensions()

    # Load the words to analyze and the test words with known political sensitivity
    words_to_analyze, test_words = load_words()

    # Find the best dataset and dimension based on the test words
    best_dataset_name, best_dim_name, min_error = find_best_dataset_dim(datasets, dims, test_words)
    print(f"Best dataset: {best_dataset_name}, Best dimension: {best_dim_name}, Minimum error: {min_error}")

    # Use the best dataset and dimension to calculate political sensitivity
    best_dataset = datasets[best_dataset_name]
    best_dimension = dims[best_dim_name]

    # Analyze each word in the list of words to analyze
    for word in words_to_analyze:
        similar_words = calculate_political_sensitivity(best_dataset, best_dimension, word)
        print(f"Words similar in political sensitivity to '{word}': {similar_words}")

if __name__ == "__main__":
    main()