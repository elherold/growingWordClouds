import pandas as pd


def joined_sensitive_word_csvs():
    """
        This function normalizes and combines data from two CSV files containing sensitive words.

    - Reads data from 'output_buzzwords_approach.csv' and 'output_dimension_approach.csv'.
    - Normalizes the 'sensitivity_score' column in both DataFrames to a 0-1 range.
    - Concatenates the normalized DataFrames.
    - Handles duplicate entries:
        - Groups data by 'similar_word'.
        - Joins values of 'input_word' (comma-separated and removing duplicates).
        - Calculates the mean 'sensitivity_score'.
    - Sorts the combined DataFrame by 'sensitivity_score' (descending).
    - Saves the processed data to 'joined_sensitive_words.csv'.
    """

    # Read data from CSV files
    df1 = pd.read_csv('output/output_buzzwords_approach.csv')
    df2 = pd.read_csv('output/output_dimension_approach.csv')

    # Normalize sensitivity score (0-1 range)
    df1['sensitivity_score'] = (df1['sensitivity_score'] - df1['sensitivity_score'].min()) / (df1['sensitivity_score'].max() - df1['sensitivity_score'].min())
    df2['sensitivity_score'] = (df2['sensitivity_score'] - df2['sensitivity_score'].min()) / (df2['sensitivity_score'].max() - df2['sensitivity_score'].min())

    # Join DataFrames
    df = pd.concat([df1, df2])

    # Handle duplicates (mean score for similar words)
    df = df.groupby('similar_word').agg({'input_word': lambda x: ', '.join(set(', '.join(x).split(', '))), 'sensitivity_score': "mean"}).reset_index()

    # Sort by sensitivity score (descending)
    df = df.sort_values('sensitivity_score', ascending=False).reset_index(drop=True)

    # Save processed data to CSV
    df.to_csv('output/joined_sensitive_words.csv', index=False)


if __name__ == "__main__":
    joined_sensitive_word_csvs()
