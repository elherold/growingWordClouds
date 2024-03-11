# Growing Word Clouds
by Jonas Jocham, Elena Herold, Jana Stefan, Lina, Lukas Weber

## Technical Setup & Execution of Prototype
This project was implemented with Python 3.12.   
Steps to run the code: 

`pip install -r requirements.txt`  - to install all necessary packages

`python main.py`                   - to get a csv file of new sensitive words

`python gpt_api.py`                - to get descriptions for all new sensitive words

## Content
### Stakeholder & Requirement Analysis    

### Datasets & Choice of Model
1. Reddit-dataset (train.csv)
2. Pre-embedded gensim models
- Begründung für dataset, Vor- u. Nachteile von verwendetem Dataset

After thorough research, we decided to use the *word2vec* and *GloVe* architecture as they provided us with a tradeoff between computational efficiency and good results.
Note: the current model is only trained on an English corpus.
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
contains all output words found and filtered according to their sensitivity, as defined by sensitive_buzzwords_approach.py

---------------------------------------------

#### dimension_evaluation.py
This file is evaluating various dimensions related to political loadedness across a selection of datasets. It identifies the optimal combination of dimension and dataset by measuring the errors against a set of test data. 
The choice of potential informative dimensions are aligned with recognized lines of political conflict (see for example:), although it is important to note that they are greatly simplified and do not fully encapsulate the complexities of real-world political ideologies. 

Initially, the file loads various datasets, including the previously defined reddit model and pre-embedded gensim word embeddings based on Twitter, Google News, and Wikipedia data. It then identifies the best performing informative dimension for each embedding space, based on the pre-defined set of possible dimensions. The script uses cosine similarity measurements between a set of test words and the calculated axes to assess the effectiveness of each dataset-dimension pairing. Finally the script displays the best performing combination and saves the dimension data to a JSON file. Be aware that the limited hand-labeled test data and the small number of dimensions and datasets compared are due to time constraints. With a larger amount of labeled test data and more datasets for comparison, there's potential for enhanced results. This file is designed as a versatile framework that can be readily adjusted for various datasets and dimensions, aiding in the identification of the most effective combinations for future applications. It's important to note that this file is not a core component of the main pipeline but rather supports the optimization of the informative_dimension_approach.py 

#### best_dimension.json
This is the file where the key-words of the best performing dimension (based on dimension_evaluation.py) are stored

---------------------------------------------
#### informative_dimension_approach.py
This file contains all the functionalities required for the *informative dimension approach*. It applies the best dataset-dimension combination determined earlier, specifically using the Reddit dataset and the "social values" informative dimension. The process involves identifying politically charged temrs similar to those in the Macht Sprache database. Rather than making directly use of single buzzwords, the file calculates an informative axis using the average vector of key buzzwords that represent the spectrum of "progressive social values" and "conservative social values". The selection of these buzzwords was a balance between representativeness for each category and their presence in our embedding space, meaning they had to be part of our used dataset. 

For each input term from Macht Sprache, the file computes the top 50 most similar terms. These terms are ranked in descending order based on their absolute cosine similarity to the informative axis. From this ranking, the top 10 temrs (i.e., those most closely aligned with one of the ends of the axis) are returned. This approach differs from the initial one as it doesn't use a fixed threshold for sensitivity over all terms. Instead, it depends on the sensitivity score distribution of the most similar temrs for a given Macht Sprache term. The parameter N=50, dictating the number of considered words for the sensitivity rating, greatly influences the output. For words frequently used on non-political contexts like "woke", a higher N value proved beneficial to ensure inclusion of politically loaded terms in the analysis. Converseley, terms with very frequent political connotations like "abortion" yield better results with a lower N. Therefore, N=50 represents a compromise between these two tendencies. 

#### sensitive_analysis.csv
contains all output words found and filtered according to their sensitivity as defined by informative_dimension_approach.py

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

