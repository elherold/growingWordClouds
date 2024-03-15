# Growing Word Clouds
by Jonas Jocham, Elena Herold, Jana Stefan, Lina, Lukas Weber

## Technical Setup & Execution of Prototype
This project was implemented with Python 3.12.   
Steps to run the code: 

`pip install -r requirements.txt`  - to install all necessary packages

`python main.py`                   - to get a csv file of new sensitive words

`python gpt_api_calls.py`          - to prolong the dictionary of descriptions for all new sensitive words. Be careful, this may cost money! The user is expected to have a look at the gpt_api_calls() function to understand the parameters and specify them according to their needs. At least: N_CALLS, START_INDEX and OUTPUT_LANGUAGE
    

## Content

### MACHT.SPRACHE.
Macht.Sprache. is an online tool that aims to support the translation of texts in a politically sensitive manner and increase the public understanding of the power that language can hold. 
Currently, the tool has a manually manipulated word list of sensitive terms that is based on “Erfahrungswissen”, knowledge based on experience of culture and language specialists. But picking terms manually is a very slow procedure that takes time and effort that the developers of Macht.Sprache. can not always invest in their tool. The incentive of this group was therefore to develop an algorithm that automatically expands the Macht.Sprache. data base to build new word clouds with possibly sensitive terms.

### WHAT IS A SENSITIVE TERM?
Before any other step can be taken, a general definition of what makes a term sensitive needs to be agreed on. In very close collaboration with students of cultural studies, such a definition was crafted: words or phrases with the potential to provoke offense, upset, or lead to controversy. It is important to note here that a term is sensitive for our tool once it is either offensive, or politically loaded. So could any term leading to a political debate be detected here, and not only those causing immediate harm. We have decided on this approach after thorough debate, since we want to deliver result optimizations to website users when facilitating translation tasks. Our analysis of the pre-existing terms on Macht.Sprache. has shown that not only harmful terms can lead to difficulties in translation, but also terms that are used in very culture-specific contexts. This issue could be solved by implementing a rating score on the website to indicate harmfulness of terms from affected induviduals. This, however, is outside the scope of our word cloud task, but has been implemented by another project group. 

### STAKEHOLDERS
In the requirement analysis, several stakeholders were defined. Of immediate importance are the website users. As a translation tool, an urgent requirement is that the presented terms are helpful in translation tasks. To increase the efficiency in this regard, the idea to include word descriptions came up. Including those descriptions will not only lead to an immediate understanding of a term, but can also deliver an explanation of why a certain word needs to be handled with increased attention. 
These word descriptions may also prove useful for another stakeholder: the crowd source project. In the seminar, unfortunately no group decided to work on this specific project. But any final control instance that makes the final decision on which word of our proposed words makes it to the Macht.Sprache. website could benefit from a very short word description. Further, an automatically calculated sensitivity score could ease the work of the crowd sourcers. Of general interest is a high correctness of the words. 
For us as the project developers, a realizable goal is of essence, paired with an appropriate balance of the short time and a wish for high quality. 
When dealing with sensitive language, it is indispensable to handle harmful terms in a careful, respectful manner to avoid causing harm for affected groups and individuals. Including heterogenous data is an effective way to avoid bias and objectify a model. 

### DATA SETS
The used data set is going to influence the outcome of our model. When using a biased data set, the model can be expected to deliver biased output. When using sensitively written data, the model can be expected to lack harmful terms, the ones that we are explicitly searching for. Bridging the urge to use ethical data and still be able to provide Macht.Sprache. with sensitive terms is a challenging incentive. After looking at a vast variety of potential data sets, from parliament speeches over hip hop lyrics to lexica and novels, we have decided to train our model on a data set consisting of Reddit comments. 
Comments from a public discussion platform stem by nature from a very heterogenic environment. People from all kinds of cultural and political backgrounds share their thoughts on a broad variety of topics, often enough in a tonality that is lacking respect for each other. While harmful communication is an issue, we decided to instrumentalize this to our advance, since lack of respect provides us with the exact terms that we are looking for.

### TECHNICAL REQUIREMENTS
To implement the interests of the identified stakeholders, several technical steps are needed. Here, the basic structure of the algorithm is shortly introduced. More detail can be found in the commented code files.
Data from the chosen set travels through several steps, changing to some level in each single one. Firstly, the data set is used to train a word embedding tool. We have decided for Word2Vec as our default as it provides us with a tradeoff between computational efficiency and good results. Note: the current model is only trained on an English corpus. We pick a term from the Macht.Sprache. website that we want to find semantically similar words to and measure the distance in the word embedding via cosine similarity. A list of similar words is being created. To calculate their sensitivity scores, we designed two approaches, one measuring the similarity to pre-defined sensitive buzzwords, the other one tweaking the words into informative dimensions. The most sensitive words are now being forwarded to an individualized GPT-4 chatbot that is tasked to write very short, informative descriptions for each term and label them with a sensitivity score in between 0 (not sensitive) and 1 (highly sensitive). 



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
This file contains all the functionalities required for the *informative dimension approach*. It applies the best dataset-dimension combination determined earlier, specifically using the Reddit dataset and the "social values" informative dimension. The process involves identifying politically charged terms similar to those in the Macht Sprache database. Rather than making directly use of single buzzwords, the file calculates an informative axis using the average vector of key buzzwords that represent the spectrum of "progressive social values" and "conservative social values". The selection of these buzzwords was a balance between representativeness for each category and their presence in our embedding space, meaning they had to be part of our used dataset. 

For each input term from Macht Sprache, the file computes the top 50 most similar terms. These terms are ranked in descending order based on their absolute cosine similarity to the informative axis. From this ranking, the top 10 terms (i.e., those most closely aligned with one of the ends of the axis) are returned. This approach differs from the initial one as it doesn't use a fixed threshold for sensitivity over all terms. Instead, it depends on the sensitivity score distribution of the most similar terms for a given Macht Sprache term. The parameter N=50, dictating the number of considered words for the sensitivity rating, greatly influences the output. For words frequently used on non-political contexts like "woke", a higher N value proved beneficial to ensure inclusion of politically loaded terms in the analysis. Converseley, terms with very frequent political connotations like "abortion" yield better results with a lower N. Therefore, N=50 represents a compromise between these two tendencies. 

#### sensitive_analysis.csv
contains all output words found and filtered according to their sensitivity as defined by informative_dimension_approach.py

---------------------------------------------
### join_csvs.py
This file contains the functionality to join the output of the buzzwords approach and the informative dimension approach. It takes the two CSV files and returns a list of new sensitive words with their combined sensitivity score. This list is then saved in a CSV file.

---------------------------------------------
#### gpt_api_calls.py
This file calls the OpenAI API chat completion models (so far we used GPT3.5 and GPT 4, may also be used with successor models). It takes a list of words (CSV file with the columns similar_word, input word) and returns a dictionary of the words with GPT generated sensitivity score, a definition, and 4 translation options and their respective nuance. This dictionary is then saved in a JSON file.

To use the OpenAI API, you need to have an API key from OpenAI. This key needs to be stored in a file other than that empty file called "API_KEY" in the same directory as the gpt_api_calls.py file. In the current setup, 5 word descriptions consume 1500 tokens and cost ~0.05$.

In both the English and German prompt files, the system prompt used for the GPT API calls is stored. The prompt is used to generate the sensitivity score and the definition of the words. The GPT API is called with the prompt and the word to be analyzed. We set the "temperature" parameter of the model low so that the output is consistent. The output is then parsed and stored in a dictionary.	

To not waste resources the number of words we requested the API so far is limited to 160 for both English and German. The file can easily be used to produce more word descriptions if the content is considered valuable. Also, the system prompt can be changed to produce different outputs. This leaves room for further development and improvement of the system.

#### english_prompt.txt
The system prompt is used for the GPT API calls if output is desired in English. It is used to describe as accurately as possible what the expected output looks like.
The first paragraph describes the general task.
The second and third describe the principles of macht.sprache.
In the fourth paragraph the JSON format and the keys of the response are specified, so that the response can be parsed correctly.

#### german_prompt.txt
The system prompt is used for the GPT API calls if output is desired in German. Its structure is equal to the English prompt.

#### gpt_descriptions_english.json
contains the output of the GPT API calls for the English language.
"word", "sensitivity_rating", "definition" and "translation_options" are the keys of the dictionary. The values are the words, the sensitivity rating, the definition and the translation options of the words. The translation options are a list of dictionaries with the keys "option" and "nuance". The value of "option" is the translation of the word and the value of "nuance" is the nuance of the translation.

#### gpt_descriptions_german.json
contains the output of the GPT API calls for the German language. The structure is equal to the English JSON, just the language of values is German.

#### tokens_used.csv
Tracks tokens to overview costs.
Reset this file to track the tokens used.

---------------------------------------------


### EVALUATION
How well does the model perform?
Gathering terms within a certain scope of similarity to previously as sensitive defined words works out very well. Following the definition of sensitivity as defined earlier, that is not only harmful and insulting but also politically loaded, the vast majority of newly filtered terms can be labeled as sensitive. Keeping in mind that another instance (eg crowd sourcers) is tasked to deliver a final judgement on which terms makes it to Macht.Sprache. and which ones do not, a satisfyingly high correctness has been achieved. In this regard, the original demand to build word clouds with sensitive terms has been successfully realized. Furthermore, features to define the degree of sensitivity, as well as to describe each term have been implemented. This was done in order to ease the work load of the crowd sourcers as one of our previously defined stakeholders. Our algorithm can provide translators and website users with a vast amount of words, not in order to prohibit certain language use, but in order to support careful reflection on the power and effect of used terms. Not only can terms be defined in English, but also in German, which is of essential interest for translation tasks. 
In a final wrap up, the project’s incentive was realizable in the short amount of time available. Us ourselves were surprised by some of the precise definitions delivered by the chat bot as well as the high degree of correctness achieved. The stakeholders requirements were all implemented. 
One thing still left to debate is the used data set. While for our prototype, Reddit comments were more than sufficient, it is up to experiments to find out whether another data set may lead to an even more precise outcome. 

### CRITICAL THOUGHTS
Apart from data sets, large language models are known to fall prey to certain biases. Once one link carries a certain bias, the entire chain may produce a biased output. The actual impact of this issue is yet to be explored and currently increasingly debated in society and amongst technology experts – for example in context to Google’s lates image production tool Gemini. 
Further, the current data set does not include all terms available on Macht.Sprache. This issue could be fixed in training the model on more or bigger data sets. Another issue, however, is the inability of our tool to filter for terms existing out of two or more words. In a future design, this is an important feature, since otherwise terms like “racial profiling” and “white supremacy” cannot be identified.


---------------------------------------------

Link to our Miro Board, where you can find our brainstorming and research visualization as well as our presentation: 
https://miro.com/app/board/uXjVNplZDZk=/?share_link_id=253751004358

---------------------------------------------



