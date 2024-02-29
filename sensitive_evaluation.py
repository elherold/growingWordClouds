import gensim.downloader as api
import numpy as np
import os
import pickle
from gensim.models import KeyedVectors 
import json

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
    word_vector = embedding_space[word]
    projection = cosine_similarity(word_vector, axis)
    return projection

def find_best_dataset_dim(datasets, dims, test_words):
    """
    Identify the best dataset-dimension combination for identifying political sensitivity,
    based on the lowest average absolute error in projections compared to expected sensitivity labels.
    
    Args:
        datasets:list of dataset objects for evaluation.
        dims: list of dicts, each containing "positive" and "negative" keys with word lists defining political axes for each dimension.
        test_words: dict of test words with labels indicating political sensitivity (1) or neutrality (0).
        
    Returns:
        A tuple containing the name of the best dataset, the name/number of the best dimension, and the lowest average absolute error.
    """
    min_error = np.inf
    best_dataset = None
    best_dim = None



    for dataset in datasets:
        # Check if the dataset is a Word2Vec model and get its KeyedVectors
        vectors = dataset.wv if hasattr(dataset, 'wv') else dataset

        for dim_name, dim_values in dims.items():
            pos_words = dim_values["positive"]
            neg_words = dim_values["negative"]
            total_error = 0
            num_words_evaluated = 0
            axis = create_vec_axis(vectors, pos_words, neg_words)
            
            for word, label in test_words.items():
                if word not in vectors.key_to_index:
                    print(f"The word {word} from the dimension {dim_name} is missing from the dataset {str(dataset)}")
                    continue  # Skip words not in the dataset
                projection = project_word_on_vec(vectors, word, axis)
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
                    best_dataset = str(dataset)
                    best_dim = dim_name

    return best_dataset, best_dim, min_error

def load_embeddings(models_dir="models"):
    embeddings_list = []
    
    for filename in os.listdir(models_dir):
        file_path = os.path.join(models_dir, filename)
        
        if filename.endswith(".pkl"):
            # Handle pickle files
            with open(file_path, "rb") as f:
                model = pickle.load(f)
                embeddings_list.append(model)
                print(f"Loaded pickle model from {file_path}")
                
        elif filename.endswith(".model"):
            # Handle model files (assuming they are gensim models for this example)
            model = KeyedVectors.load(file_path, mmap='r')  # Use the appropriate load function for your model type
            embeddings_list.append(model)
            print(f"Loaded gensim model from {file_path}")
    
    return embeddings_list


def define_political_dimensions():
    dims = {
       "economic":{"positive":["socialism", "welfare", "equality", "redistribution", "taxes", "healthcare", "universal", "subsidies", "cooperative"], "negative":["capitalism", "deregulation", "privatization", "markets", "taxcuts", "insurance",  "monopoly", "inequity", "exploitation"]},
        "social": {"positive":[ "equality", "rights", "feminism", "queer", "diversity", "reform", "inclusion", "justice", "empowerment", "tolerance"], "negative":["tradition", "patriotism", "nationalism", "family", "heritage", "order", "conservatism", "segregation", "exclusion", "inequality"]},
        "environment": {"positive":["climate", "renewable", "conservation", "sustainable", "green", "ecology","biophilia", "restoration", "permaculture", "biodiversity"], "negative":["coal", "drilling", "deregulation", "growth", "nuclear", "oil","deforestation", "pollution", "extinction", "waste"]},
        "foreign": {"positive":["cooperation", "rights", "globalization", "NATO", "trade", "peace","diplomacy", "multilateralism", "aid", "openness"],"negative":["sovereignty", "borders", "tariffs", "nationalism", "security", "immigration","isolationism", "protectionism", "conflict", "xenophobia"]},
        "governance": {"positive":["democracy", "transparency", "liberty", "rights", "press", "justice","accountability", "participation", "equality", "rule of law"], "negative":["power", "surveillance", "control", "censorship", "state", "security","corruption", "authoritarianism", "inequality", "impunity"]}
    }
    return dims

def load_words():
    test_words = {
        # Politically Loaded Terms (1)
        "fascism": 1,"socialism": 1,"impeachment": 1,"referendum": 1,"nationalism": 1,"abortion": 1,"censorship": 1,"sanctions": 1,"tariffs": 1,"protest": 1,
    
        # Politically Neutral Terms (0)
        "library": 0,"restaurant": 0,"mountain": 0,"ocean": 0,"piano": 0,"calendar": 0,"umbrella": 0,"street": 0,"plate": 0,"window": 0
    }
    return  test_words

def write_best_dimension_to_json(best_dim, dims):
    # Extract the data for the best dimension
    best_dim_data = dims[best_dim]
    
    # Specify the filename
    filename = 'best_dimension.json'
    
    # Write the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(best_dim_data, f, indent=4)
    
    print(f"Contents of the best dimension '{best_dim}' were written to {filename}")

def main():
    datasets = load_embeddings()
    dims = define_political_dimensions()
    test_words = load_words()

    best_dataset, best_dim, error = find_best_dataset_dim(datasets, dims, test_words)

    write_best_dimension_to_json(best_dim, dims)

    print(f"The dataset-dimension pair with the lowest error on the test words is {best_dataset} with the {best_dim} dimension, with an error of {error}")

if __name__ == "__main__":
    main()