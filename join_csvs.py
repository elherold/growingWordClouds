import pandas as pd
import numpy as np

def main():
    df1 = pd.read_csv('similar_sensitive_words.csv')
    df2 = pd.read_csv('sensitivity_analysis.csv')

    # Normalize the values of "sensitivity_score" column of df1 and df2 to range between 0 and 1
    df1['sensitivity_score'] = (df1['sensitivity_score'] - df1['sensitivity_score'].min()) / (df1['sensitivity_score'].max() - df1['sensitivity_score'].min())
    df2['sensitivity_score'] = (df2['sensitivity_score'] - df2['sensitivity_score'].min()) / (df2['sensitivity_score'].max() - df2['sensitivity_score'].min())
    
    # Join two DataFrames of the same structure
    df = pd.concat([df1, df2])
    
    # In case of duplicates in similar_word, join values of "input_word" and calculate mean sensitivity_score
    df = df.groupby('similar_word').agg({'input_word': lambda x: ', '.join(set(', '.join(x).split(', '))), 'sensitivity_score': "mean"}).reset_index()
    
    # Sort by sensitivity_score
    df = df.sort_values('sensitivity_score', ascending=False).reset_index(drop=True)
    
    # Store the DataFrame in a CSV file
    df.to_csv('joined_sensitive_words.csv', index=False)

if __name__ == "__main__":
    main()
