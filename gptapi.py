import pandas as pd
from openai import OpenAI
import json

def read_api_key(filepath):
    """
    Reads the OpenAI API key from a file.
    
    :param filepath: The path to the file containing the API key.
    :return: The API key as a string.
    """
    with open(filepath, 'r') as file:
        return file.read().strip()

def load_data_and_prepare_requests(file_name, chunks=4):
    """
    Loads and Prepares the data for API requests based on DataFrame chunks.
    
    :param DATA_FILE_NAME: A string of a csv file with the columns: "similar_word" (str), "input_word" (str) and "sensitivity_score" (float).
    :param CHUNKS: The number of chunks to split the DataFrame into.
    :return: A list of prepared requests.
    """
    df = pd.read_csv(file_name)
    df_chunks = [df[i:i + n] for i in range(0, len(df), n)]
    requests = [
        [
            ("word: " + row["similar_word"], "word_cloud_reference: " + row["input_word"])
            for _, row in df_chunk.iterrows()
        ]
        for df_chunk in df_chunks[:CHUNKS]  # Limit to first 4 chunks to be mindful of token limits
    ]
    return requests

def send_request(client, requests, prompt, model, file_name):
    """
    Sends requests to the OpenAI API and writes responses to a file.
    
    :param client: The OpenAI client.
    :param requests: The list of prepared requests.
    :param prompt: The prompt to use for the requests.
    :param model: The model to use for the requests.
    :param file_name: The name of the file to write the responses to.
    :return: The total number of tokens used by the requests.
    """
    tok = 0
    for request in requests:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{request}"},
            ],
            temperature=0.1,
        )
        process_response(completion, file_name)
        tok += completion.usage.total_tokens
    return tok

def process_response(completion, file_name):
    """
    Processes the API response and writes it to a file.
    
    :param completion: The completion object from the API response.
    :param file_name: The file to which the response should be written.
    """
    txt = "[" + completion.choices[0].message.content.split('[', 1)[1]
    parts = txt.split(']')
    output = ']'.join(parts[:-1]) + ']'

    try:
        # Read the existing JSON file
        with open(file_name, 'r') as file:
            json_data = json.load(file)

        new_data = json.loads(output)

        # Combine the data
        json_data.extend(new_data)  # You can use list concatenation with `+` if you prefer
    except:
        json_data = json.loads(output)

    # Write the combined list back to the original file (or a new file, if desired)
    with open(file_name, 'w') as file:
        json.dump(json_data, file, indent=4)  # `indent` for pretty printing, optional



def main():
    """
    Main function to orchestrate the execution of the script.
    """
    API_KEY_FILE = 'API_KEY'
    CHUNKS = None
    OUTPUT_FILE_NAME = 'output_english.json'
    DATA_FILE_NAME = 'joined_sensitive_words.csv'
    OUTPUT_LANGUAGE = 'english' # or 'german'
    INPUT_LANGUAGE = 'English' if OUTPUT_LANGUAGE == 'english' else 'englisch' # or 'German' and 'deutsch'
    
    api_key = read_api_key(API_KEY_FILE)
    client = OpenAI(api_key=api_key)
    model = "gpt-4-turbo-preview"  # or another model like "gpt-3.5-turbo"
    # read the system prompt from the file english_prompt.txt
    sys_prompt_english = open('english_prompt.txt', 'r').read()
    sys_prompt_german = open('german_prompt.txt', 'r').read()
    sys_prompt = sys_prompt_english if OUTPUT_LANGUAGE == 'english' else sys_prompt_german

    requests = load_data_and_prepare_requests(DATA_FILE_NAME)
    total_tokens = send_request(client, requests, sys_prompt, model, OUTPUT_FILE_NAME)

    print(f"Total tokens used: {total_tokens}")
    with open('tokens_used.csv', 'a') as file:
        file.write(f"{total_tokens}\n")

if __name__ == "__main__":
    main()
