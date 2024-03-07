# growingWordClouds

## Technical Setup & Execution of Prototype
### requirements.txt
This project was implemented with Python 3.12.
Steps to run the code: 

pip install -r requirements.txt

run main.py

run gpt_api.py

## Content
### Stakeholder & Requirement Analysis    

### Datasets & Choice of Model
1. Reddit-dataset (train.csv)
2. Pre-embedded gensim models
- Begründung für dataset, Vor- u. Nachteile von verwendetem Dataset

After thorough research, we decided to use the *word2vec* and *GloVe* architecture as they provided us with a tradeoff between computational efficiency and good results.
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
First, a list of similar words is calculated for each word in the macht.sprache database. As a similarity measure, we decided to use the cosine distance as it exclusively focuses on the meanings of the words and leaves out their frequency, which means that also rarely used words are included. Only the similar words that exceed a similarity threshold are kept. Here, a threshold of 0.6 showed the best results.      
A sensitivity value is then calculated for each new word. The idea behind this approach is that the sensitivity of words can be inferred based on their similarity to certain social justice buzzwords. We chose the buzzwords 'discrimination' and 'political'as we saw them as the smallest set of commonalities between all the concepts in the social justice debate. Then, the cosine distance is calculated between each new word and the two buzzwords. We hypothesized that the closer the new term is to these buzzwords, the more its connotation is linked to political sensitivity.   
Only words that exceed a sensitivity threshold are retained and then ranked according to their sensitivity score. Here, a threshold of 0.4 showed the best results.

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
#### gptapi.py (umbenennen zu gpt_api.py)
#### english.prompt.txt
#### german.prompt.txt
#### output.json (umbenennen!)
---------------------------------------------

---------------------------------------------
Link to our Miro Board, where you can find our brainstorming and research visualization as well as our presentation: 
https://miro.com/app/board/uXjVNplZDZk=/?share_link_id=253751004358

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

