import pandas as pd
from openai import OpenAI

def initialize_client(api_key_file='API_KEY'):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    return OpenAI(api_key=api_key)

def chunk_dataframe(df, chunk_size=5):
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def prepare_requests(df_chunks):
    requests = []
    for df_chunk in df_chunks[:4]:  # Limiting to the first 4 chunks to be careful of token limits
        chunk_requests = [
            ("word: " + row["similar_word"], "word_cloud_reference: " + row["input_word"])
            for _, row in df_chunk.iterrows()
        ]
        requests.append(chunk_requests)
    return requests

def send_requests(client, requests, prompt, model="gpt-4-turbo-preview", output_file="output.json"):
    total_tokens = 0
    for request in requests:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{request}"},
            ],
            temperature=0.1,
        )
        # Process the response and append it to the output file
        process_response(response, output_file)
        total_tokens += response.usage.total_tokens
    return total_tokens

def process_response(response, output_file):
    # Example response processing, needs to be adapted based on actual needs
    content = response.choices[0].message.content
    with open(output_file, 'a') as file:
        file.write(content + '\n')

def main():
    client = initialize_client()
    # Example DataFrame loading or creation
    # df = pd.DataFrame({'similar_word': ['word1', 'word2'], 'input_word': ['input1', 'input2']})
    # df_chunks = chunk_dataframe(df)
    # requests = prepare_requests(df_chunks)
    # Assuming sys_prompt_german is defined as shown above
    # total_tokens = send_requests(client, requests, sys_prompt_german)
    # print(f"Total tokens used: {total_tokens}")

if __name__ == "__main__":
    main()
