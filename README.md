## Home_Assignment_4
## Student Name : Sai krishna Edara
## Student Id : 700769262
## Question 1
### NLP Preprocessing Script
#### Overview
This Python script demonstrates a basic Natural Language Processing (NLP) preprocessing pipeline using the NLTK library. It processes a sentence by performing three steps:

##### Tokenization:
Splitting the sentence into individual words and punctuation.
##### Stopword Removal:
Removing common words (e.g., "and", "the") that carry little meaning.
##### Stemming:
Reducing words to their root form (e.g., "running" → "run").

The script is useful for preparing text data for tasks like text analysis, sentiment analysis, or machine learning.
#### Requirements

Python 3.x
NLTK library (pip install nltk)

The script automatically downloads required NLTK resources (punkt, stopwords, punkt_tab) if they are not already installed.
#### Usage

##### Install Dependencies:
pip install nltk


##### Run the Script:
python preprocess.py

The script processes a sample sentence:"NLP techniques are used in virtual assistants like Alexa and Siri."

##### Output:
The script prints the results of each preprocessing step:

Original tokens
Tokens without stopwords
Stemmed words

#### Customization

Change the Input Sentence: Modify the sentence variable in the script to process a different sentence.
Extend Functionality: Add more preprocessing steps (e.g., lemmatization, part-of-speech tagging) using NLTK's tools.

#### Notes

The averaged_perceptron_tagger resource is downloaded but not used in this script. It is included for potential future extensions (e.g., POS tagging).
The Porter Stemmer is a simple stemming algorithm. For more accurate results, consider using NLTK's WordNet Lemmatizer.

## Question 2
### Named Entity Recognition (NER) Script
#### Overview
This Python script uses the spaCy library to perform Named Entity Recognition (NER) on a given sentence. It identifies and extracts named entities such as people, organizations, locations, dates, and events, along with their types and positions in the text.
The script is useful for extracting structured information from unstructured text, with applications in information extraction, text summarization, and knowledge graph construction.
#### Requirements

Python 3.x
spaCy library (pip install spacy)
spaCy English language model (en_core_web_sm)

#### Install the requirements using:
pip install spacy
python -m spacy download en_core_web_sm

#### Usage

Install Dependencies:
pip install spacy
python -m spacy download en_core_web_sm


#### Run the Script:
python ner_script.py

The script processes a sample sentence:"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

#### Output:
The script prints details for each detected named entity:

Entity text
Entity label (e.g., PERSON, GPE, DATE)
Start and end character positions in the sentence


#### Customization

#### Change the Input Sentence:
Modify the sentence variable in the script to process a different sentence.
#### Use a Different Model:
Replace "en_core_web_sm" with a larger model (e.g., en_core_web_md or en_core_web_lg) for potentially better accuracy (requires downloading the model).
#### Extend Functionality:
Add additional spaCy features like part-of-speech tagging or dependency parsing by accessing other attributes of the doc object.

#### Notes

The en_core_web_sm model is lightweight but may have lower accuracy compared to larger models like en_core_web_lg.
Ensure the spaCy model is downloaded before running the script, as it is not included in the spaCy library by default.
Entity labels depend on the model. Common labels include:
PERSON: People (e.g., "Barack Obama")
GPE: Geopolitical entities (e.g., "the United States")
DATE: Dates or time periods (e.g., "2009")
EVENT: Named events (e.g., "the Nobel Peace Prize")

## Question 3

### Scaled Dot-Product Attention Script
#### Overview
This Python script implements the Scaled Dot-Product Attention mechanism, a core component of the Transformer model introduced in "Attention is All You Need" (Vaswani et al., 2017). The script computes attention weights based on queries (Q), keys (K), and values (V) and produces a weighted output, using PyTorch for tensor operations and NumPy for input/output handling.
The script is useful for understanding the attention mechanism used in Transformer-based models for tasks like machine translation, text generation, and more.
#### Requirements

Python 3.x
NumPy (pip install numpy)
PyTorch (pip install torch)

#### Install the requirements using:
pip install numpy torch

#### Usage

##### Install Dependencies:
pip install numpy torch


##### Run the Script:
python attention_script.py

The script processes sample input matrices:

Queries (Q): [[1, 0, 1, 0], [0, 1, 0, 1]]
Keys (K): [[1, 0, 1, 0], [0, 1, 0, 1]]
Values (V): [[1, 2, 3, 4], [5, 6, 7, 8]]


#### Output:
##### The script prints:

Attention Weights: The softmax-normalized attention scores.
Output Matrix: The final output, a weighted combination of the value vectors.


#### Customization

##### Change Input Matrices:
Modify the Q, K, and V matrices in the script to experiment with different inputs.
##### Add Batch Dimension:
Extend the inputs to include a batch dimension (e.g., (batch_size, seq_len, d_k)) for processing multiple sequences.
##### Incorporate Masks:
Add attention masks to handle padding or causal attention (e.g., for autoregressive models).

#### Notes

The script assumes Q, K, and V have compatible shapes: Q and K must have the same last dimension (d_k), and V must have the same sequence length as K.
The scaling factor (sqrt(d_k)) prevents large dot-product values, stabilizing gradients during training.
This is a simplified implementation. Real-world Transformer models use multi-head attention and additional components like feed-forward layers.

## Question 4
### Sentiment Analysis Script
#### Overview
This Python script uses the Hugging Face Transformers library to perform sentiment analysis on a given sentence. It leverages a pre-trained model to classify the sentiment as positive or negative and provides a confidence score for the prediction.
The script is useful for analyzing the sentiment of text data in applications like customer feedback analysis, social media monitoring, or product review processing.
#### Requirements

Python 3.x
Transformers library (pip install transformers)
PyTorch or TensorFlow (automatically installed with Transformers, depending on your setup)

#### Install the requirements using:
pip install transformers

#### Usage

Install Dependencies:
pip install transformers


#### Run the Script:
python sentiment_analysis.py

The script processes a sample sentence:"Despite the high price, the performance of the new MacBook is outstanding."

#### Output:
##### The script prints:

Sentiment: The predicted sentiment label ("POSITIVE" or "NEGATIVE").
Confidence Score: The model's confidence in the prediction (between 0 and 1).

#### Customization

#### Change the Input Sentence:
Modify the sentence variable to analyze a different sentence.
#### Use a Specific Model:
Specify a different pre-trained model by passing the model name to the pipeline, e.g., pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment").
#### Batch Processing:
Pass a list of sentences to classifier (e.g., classifier([sentence1, sentence2])) to analyze multiple sentences at once.

#### Notes

The default model (distilbert-base-uncased-finetuned-sst-2-english) is optimized for English binary sentiment classification (positive/negative).
For non-English text or multi-class sentiment (e.g., 1-5 stars), use a different model suited to the task.
The pipeline automatically downloads the pre-trained model and tokenizer the first time it is run, which requires an internet connection.
Ensure you have either PyTorch or TensorFlow installed, as the Transformers library relies

