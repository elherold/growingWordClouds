import pandas as pd
from openai import OpenAI
import json
import time

def read_api_key(file_path):
    """
    Reads the OpenAI API key from a file.
    Args:
        filepath (str): The path to the file containing the API key.

    Returns:
        str: The API key as a string.

    """
    with open(file_path, 'r') as file:
        return file.read().strip()

def load_data_and_prepare_requests(file_path, batchsize, timestamp_start, n_calls=4, start_index=0):
    """
    Loads and Prepares the data for API requests based on DataFrame chunks.
    
    Args:
        file_name (str): The csv file with the columns: "similar_word" (str), "input_word" (str) and "sensitivity_score" (float).
        batchsize (int): The number of words that one call to the api should contain.
        n_calls (int): The number of calls to the API.
    
    Returns:
        list: A list of prepared string requests.
    """
    pre_requests = time.time()
    print("prerequests: " + str(pre_requests - timestamp_start))

    df = pd.read_csv(file_path)
    df = df.iloc[start_index:]
    df_chunks = [df[i:i + batchsize] for i in range(0, len(df), batchsize)]
    requests = [
        [
            ("word: " + row["similar_word"], "word_cloud_reference: " + row["input_word"])
            for _, row in df_chunk.iterrows()
        ]
        for df_chunk in df_chunks[:n_calls]  # Limit to first 4 chunks to be mindful of token limits
    ]

    post_requests = time.time()
    print("postrequests: " + str(post_requests - pre_requests))

    return requests

def send_request(client, request, prompt, model):
    """
    Sends requests to the OpenAI API and writes responses to a file.
    
    Args:
        client (openai.OpenAI): The OpenAI client.
        requests (list of str): The list of prepared requests.
        prompt (str): The system prompt to use for the requests.
        model (str): specifies which OpenAI model to use for the requests.
        file_name (str): The name of the file to write the responses to.

    Returns:
        int: The total number of tokens used by the requests.
    """
    pre_chat = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{request}"},
        ],
        temperature=0.1,
    )

    post_chat = time.time()
    print("post api call: " + str(post_chat - pre_chat))
    return completion.choices[0].message.content, completion.usage.total_tokens, post_chat

def write_response(content, file_name, post_chat):
    """
    Processes the API response and writes it to a file.
    
    Args:
        completion (OpenAI): The completion object from the API response.
        file_name: The file to which the response should be written.
    """
    construct = '[' + str(content).split('[', 1)[1]
    parts = construct.split(']')
    json_str = ']'.join(parts[:-1]) + ']'
    try:
        with open(file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, we'll create a new list.
        file_data = []
    # call json_str as json object
    new_data = json.loads(json_str)
    # Join new_data with file_data
    file_data.extend(new_data)
    # Write the updated data to the file.
    with open(file_name, 'w') as file:
        # convert back to json.
        json.dump(file_data, file, indent=4)

    post_write = time.time()
    print("postwrite: " + str(post_write - post_chat))
    return None
        

def gpt_api_calls():
    """
    Executes the GPT API calls for generating definitions and possible translation options to a list of sensitive terms.

    This function performs the following steps:
    1. Initialize the OpenAI client.
    2. Load the system prompt from a file.
    3. Load the data and prepare requests for the API.
    4. Loop through the requests, send them to the OpenAI API, and write the responses to a file.
    5. Print the number of tokens used and writes it to a file to keep track of costs.

    The output of this function is a JSON file, containing descriptions and possible translations of the input sensitive terms.
    
    """
    ### Parameters that should be adjusted ###

    # Call parameters
    BATCHSIZE = 5 # Number of words per call
    N_CALLS = 2 # Number of calls to the API (each call will contain BATCHSIZE words)
    START_INDEX = 150 # Index of the first word to be processed

    OUTPUT_LANGUAGE = 'english' # 'german' or 'english'
    INPUT_LANGUAGE = 'English' if OUTPUT_LANGUAGE == 'english' else 'englisch' # or 'German' if OUTPUT_LANGUAGE == 'german' else 'deutsch'
    
    API_KEY_FILE = 'API_KEY' # The filename containing the OpenAI API key
    MODEL = "gpt-4-0125-preview"  # or another model like "gpt-3.5-turbo". "gpt-4-turbo-preview" points to latest version of gpt-4

    # Adjust if needed
    data_file_name = 'output/joined_sensitive_words.csv'
    output_file_name = 'output/gpt_descriptions_german.json' if OUTPUT_LANGUAGE == 'german' else 'output/gpt_descriptions_english.json'


    ### Start of the function ###
    # Time the function to get some feedback during execution
    timestamp_start = time.time()

    # Initialize the OpenAI client
    api_key = read_api_key(API_KEY_FILE)
    client = OpenAI(api_key=api_key)

    # read the system prompt from the file english_prompt.txt
    sys_prompt_english = open('output/english_prompt.txt', 'r').read()
    sys_prompt_german = open('output/german_prompt.txt', 'r').read()
    sys_prompt = sys_prompt_english if OUTPUT_LANGUAGE == 'english' else sys_prompt_german
    

    # Load the data and prepare requests for the API
    requests = load_data_and_prepare_requests(data_file_name, BATCHSIZE, timestamp_start, n_calls=N_CALLS, start_index=START_INDEX)
    
    total_tokens = 0
    # Send requests to the OpenAI API and write responses to a file
    for request in requests:
        content, toks, post_chat = send_request(client, requests, sys_prompt, MODEL)
        write_response(content, output_file_name, post_chat)
        total_tokens += toks

    print(f"Total tokens used: {total_tokens}")
    with open('util/tokens_used.csv', 'a') as file:
        file.write(f"{total_tokens}\n")

if __name__ == "__main__":
    gpt_api_calls()
