import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Function reads the input file and returns the dataframe
def read_file(path):
    df = pd.read_excel(path)
    return df

#Function lemmatizes the word tokens and returns a list of lemmatized token words
def lemmation(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_token_list = []
    for w in tokens:
        lem_token_list.append(wordnet_lemmatizer.lemmatize(w))
    return lem_token_list

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
        tokens_list.append(pos_tag(tokens))
    return tokens_list

#Function prints list passed to it
def print_list(tokens_list):
    for i in range(len(tokens_list)):
        print(tokens_list[i])


def main():
    path = input("Please enter the file path: ")
    df = read_file(path)
    processed_text = pre_process(df['details'])
    print_list(processed_text)


if __name__ == '__main__':
    main()