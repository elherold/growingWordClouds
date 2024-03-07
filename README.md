# growingWordClouds

## Stakeholder & Requirement Analysis

## Technical Setup & execution of prototype
### requirements.txt

## Content
### Datasets
1. Reddit-dataset (train.csv)
2. Pre-embedded gensim models
- Begründung für dataset, Vor- u. Nachteile von verwendetem Dataset
### Description of Files 

---------------------------------------------
#### main.py 
calls the functionalities of both the *buzzwords approach* and the *dimension approach* and fuses the output lists of new sensitive words into one list. The sensitivity scores of the new words are combined.
#### macht.sprache_words.json (input)
are the input words from the macht.sprache database.
#### word2vec_test.model
is the model used to find similar words to the input words.
#### joined_sensitive_words.csv (output)
contains the list of new sensitive words along with their combined sensitivity score and their input words.

---------------------------------------------

---------------------------------------------
#### sensitive_buzzwords_approach.py
contains all the functionalities required for the *social justice buzzwords approach*.   
First, a list of similar words is calculated for each word in the macht.sprache database. Only the similar words that exceed a similarity threshold are kept. In our analysis, a similarity threshold of 0.6 proved to be appropriate.   
A sensitivity value is then calculated for each new word. The idea behind this approach is that the sensitivity of words can be inferred based on their similarity to certain social justice buzzwords. In our analysis, the two keywords "discrimination" and "political" proved to be useful.   
Only words that exceed a sensitivity threshold are retained and then ranked according to their sensitivity score. In our analysis, a sensitivity threshold of 0.4 proved to be appropriate.

#### output_buzzwords_approach.csv
contains all new words found and filtered according to their sensitivity.

---------------------------------------------

---------------------------------------------
#### sensitive_evaluation.py
#### best_dimension.json
---------------------------------------------

---------------------------------------------
#### sensitive_cloud.py
#### sensitive_analysis.csv
#### sensitive_analysis.csv (umbenennen!)
---------------------------------------------

---------------------------------------------
#### gptapi.py 
#### english.prompt.txt
#### german.prompt.txt
#### output.json (umbenennen!)
---------------------------------------------

---------------------------------------------
# Old branch: 
gpt.api.ipynb
join_csvs.py
microsoft_news.ipynb
output_english.json 
political_sensitivity_analysis.csv
reddit_cloud.ipynb
similar_wordswith_similarity_value
social_justice_similarity.ipynb

