import gensim.downloader as api
import numpy as np
import csv
from sensitive_evaluation import project_word_on_vec, create_vec_axis
import os
import pickle
from gensim.models import KeyedVectors 
import json
import pandas as pd

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
    dataset = dataset.wv if hasattr(dataset, "wv") else dataset
    
    # Ensure the sensitive word is in the dataset
    if sensitive_word not in dataset.key_to_index:
        print(f"The word {sensitive_word} is not in the dataset.")
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
        word_projections.append([word, projection])

    
    # Order the words by their projection score (descending)
    word_projections.sort(key=lambda x: x[1], reverse=True)

    # Return the top 10 words
    return word_projections[:10]

def load_embeddings(name, models_dir="models"):
    model = None
   
    for filename in os.listdir(models_dir):
        if name==filename:
            file_path = os.path.join(models_dir, filename)
            
            try: 
                if filename.endswith(".pkl"):
                    # Handle pickle files
                    with open(file_path, "rb") as f:
                        model = pickle.load(f)
                        print(f"Loaded pickle model from {file_path}")
                        
                elif filename.endswith(".model"):
                    # Handle model files (assuming they are gensim models for this example)
                    model = KeyedVectors.load(file_path, mmap='r')  # Use the appropriate load function for your model type
                    print(f"Loaded gensim model from {file_path}")

            except Exception as e:
                print(f"Error loading model from {file_path}: {e}")
                model = None
            
            if model is not None:
                # Model loaded successfully, no need to continue
                break
            else:
                print(f"Could not load model from {file_path}.")
    
    return model

def load_dimension_from_json(filename):
    """
    Load a dimension dictionary from a JSON file.

    Args:
        filename (str): The name of the JSON file to load.

    Returns:
        dict: The dimension dictionary loaded from the JSON file.
    """
    # Construct the path to the file (optional if it's in the same directory)
    file_path = filename

    # Open the JSON file and load its content into a dictionary
    with open(file_path, 'r', encoding='utf-8') as file:
        dimension_dict = json.load(file)

    return dimension_dict

def load_sensitive_terms(json_file, model):
    """
    Load a JSON file, extract lemma words, and check if they are in the embedding space.
    
    Args:
        json_file (str): Path to the JSON file containing the lemmas.
        model (gensim.models.keyedvectors.KeyedVectors): The embedding model to check against.
    
    Returns:
        tuple: A tuple containing two lists, (found_words, missing_words).
    """
    # Initialize lists for found and missing words
    found_words = []
    missing_words = []

    # Load the JSON data
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate through each entry in the JSON data
    for entry in data:
        lemma = entry.get('lemma', '')
        
        vocab = model.wv.key_to_index if hasattr(model, "wv") else model.key_to_index

        # Check if the lemma is in the embedding space
        if lemma in vocab:
            found_words.append(lemma)
        else:
            missing_words.append(lemma)

    return found_words, missing_words

def sensitive_dimension_approach():
    # Load pretrained word embeddings
    model = load_embeddings("word2vec_test.model")

    # Define political dimension
    dim = load_dimension_from_json("best_dimension.json")

    # Define words to analyze
    sensitive_terms, words_missing_in_model = load_sensitive_terms("macht.sprache_words.json", model)

    global_similar_words = {}

    for term in sensitive_terms:
        results = calculate_political_sensitivity(model, dim, term)
        for similar_word, sensitivity_score in results:
            if similar_word not in global_similar_words or sensitivity_score > global_similar_words[similar_word]['score']:
                global_similar_words[similar_word] = {'score': sensitivity_score, 'input_word': term}

    # Convert the global tracking dict into a list of tuples for DataFrame creation
    entries = [(word, details['score'], details['input_word']) for word, details in global_similar_words.items()]

    # Sort entries by sensitivity score in descending order
    entries.sort(key=lambda x: x[1], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(entries, columns=["similar_word", "sensitivity_score", "input_word"])

    # Save DataFrame to CSV
    df.to_csv('sensitivity_analysis.csv', index=False)
 
    print(f"format of results: {df}")

if __name__ == "__main__":
    main()