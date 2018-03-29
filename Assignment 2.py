import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from matplotlib import pyplot as plt
import csv
np.set_printoptions(threshold=np.nan)


#Function reads the input file and returns the dataframe
def read_file(path):
    df = pd.read_excel(path)
    return df

def read_csv_file(path):
    result = []
    with open(path, "r") as file:
        next(file)
        reader = csv.reader(file)

        for row in reader:
            result.append(row[1]+" "+row[2]+" "+row[3])
    return result


#Function lemmatizes the word tokens and returns a list of lemmatized token words
def lemmation(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_token_list = []
    for w in tokens:
        lem_token_list.append(wordnet_lemmatizer.lemmatize(w))
    return lem_token_list


#Function for pos_tag to include only nouns, verbs, adverbs and adjectives
def pos_text(tokens):
    pos_tag_list = []
    pos_tokens = [word for word, pos in tokens if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'
           or pos == "RB" or pos == "RBR" or pos == "RBS" or pos == "JJ"
            or pos == "VB" or pos == "VBD" or pos == "VBG" or pos == "VBN" or pos == "VBP" or pos == "VBZ")]
    pos_tag_list.append(pos_tokens)
    return pos_tag_list


#Function preprocesses text to lower case and removes standard and custom stop words
def pre_process(details_list):
    custom_stopWords = ["home", "smart", "smart-home", "homemaker", "al", "n't", "'s", ".", "'", "'ve", "!", "'m",
                 ":", "-", "e.g.", "house", "(", ")", ";"]
    stopset = set(stopwords.words('english'))
    tokens_list = []

    for i in range(len(details_list)):
        str = (details_list[i]).lower()
        tokens = word_tokenize(str)
        tokens = [w for w in tokens if not w in stopset]
        tokens = [w for w in tokens if not w in custom_stopWords]
        tokens = lemmation(tokens)
        #tokens = pos_text(tokens)
        tokens_list.append(pos_text(pos_tag(tokens)))
    return tokens_list


#Function prints list passed to it
def print_list(tokens_list):
    for i in range(len(tokens_list)):
        print(tokens_list[0])


#traverse all the documents and compute a vocabulary corpus containing unique words from all the documents
def compute_vocab_list(tokens_list):
    vocab_set = set()
    counter = 0
    for list in tokens_list:
        for sublist in list:
            for doc in sublist:
                counter +=1
                vocab_set.add(doc)
    #print(counter)
    return vocab_set


#Function to compute frequency of term in a document
def freq(term, document):
  return document.count(term)


#Function to compute term frequency of terms in each document
def compute_tf(processed_text):
    document = []
    counter = 0
    tf_doc = compute_vocab_list(processed_text)

    for i in processed_text:
        for doc in i:
            #print(doc)
            tf_vector = [freq(word, doc) for word in tf_doc]
            counter += 1
            document.append(tf_vector)

    #print(len(document[999]))
    return document


#Function for computing l2 normalization
def l2_normalizer(vect):
    denominator = np.sum([element**2 for element in vect])
    return [(element / math.sqrt(denominator)) for element in vect]


#Function for computing normalized document matrix
def compute_document_matrix(document):
    doc_term_matrix_l2_norm = []
    for vect in document:
        doc_term_matrix_l2_norm.append(l2_normalizer(vect))
    #print(np.matrix(doc_term_matrix_l2_norm).shape)
    return doc_term_matrix_l2_norm


#Function for counting number of docs having the word
def number_of_docs_with_word(word, document_list):
    document_count = 0
    for doc in document_list:
        if freq(word, doc) > 0:
            document_count += 1
    return document_count


#Function to compute the idf value for each word in the document list
def compute_idf_value(word, document_list):
    number_of_documents = len(document_list)
    df = number_of_docs_with_word(word, document_list)
    return np.log(number_of_documents / 1+df)


#Function to compute idf matrix
def compute_idf_matrix(idf_vector):
    idf_matrix = np.zeros((len(idf_vector), len(idf_vector))) #Fill in zeroes where there is no value
    np.fill_diagonal(idf_matrix, idf_vector) #Fill the values diagonally
    return idf_matrix


#Function to calculate the dot product of normalized tf and idf matrix
def compute_tf_idf(normalized_doc_matrix,idf_matrix):
    doc_term_matrix_tfidf = []
    for tf_vector in normalized_doc_matrix:
        doc_term_matrix_tfidf.append(np.dot(tf_vector, idf_matrix))

    doc_term_matrix_tfidf_l2 = []
    for tf_vector in doc_term_matrix_tfidf:
        doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
    return doc_term_matrix_tfidf_l2


def scatter_plot(df):
    feat1 = df['id'].values
    feat2 = df['details'].values
    X = np.array(list(zip(feat1, feat2)))
    plt.scatter(feat1, feat2, c='black', s=7)
    plt.show()

def main():

    path = input("Please enter the file path: ")
    #df = read_file(path)
    df = read_csv_file(path)

    processed_text = pre_process(df)
    #processed_text.pop(len(processed_text)-1) #to remove the last empty row

    vocabulary = compute_vocab_list(processed_text)
    document = compute_tf(processed_text) #list of size 1000 with 1/0 comma-seperated for each word

    normalized_doc_matrix = compute_document_matrix(document)

    # For every word in the vocabulary count documents word is present in and compute word's idf value
    idf_vector = [compute_idf_value(word, processed_text) for word in vocabulary]
    idf_matrix = compute_idf_matrix(idf_vector)

    #print(idf_vector)
    #print(idf_matrix.shape)

    doc_term_matrix_tfidf_l2 = compute_tf_idf(normalized_doc_matrix, idf_matrix)
    print(np.matrix(doc_term_matrix_tfidf_l2).shape)

    #scatter_plot(df)


if __name__ == '__main__':
    main()