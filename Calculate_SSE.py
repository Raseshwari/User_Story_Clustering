from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.spatial import distance_matrix
from collections import Counter


np.set_printoptions(threshold=np.nan)


# Function reads the input file and returns the dataframe
# def read_file(path):
#     df = pd.read_excel(path)
#     return df


def read_csv_file(path):
    result = []
    user_story_ids = []
    with open(path, "r") as file:
        next(file)
        reader = csv.reader(file)

        for row in reader:
            result.append(row[1] + " " + row[2] + " " + row[3])
            user_story_ids.append(row[0])
    return result, user_story_ids


# Function lemmatizes the word tokens and returns a list of lemmatized token words
def lemmation(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_token_list = []
    for w in tokens:
        lem_token_list.append(wordnet_lemmatizer.lemmatize(w))
    return lem_token_list


# Function for pos_tag to include only nouns, verbs, adverbs and adjectives
def pos_text(tokens):
    pos_tag_list = []
    pos_tokens = [word for word, pos in tokens if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'
                                                   or pos == "RB" or pos == "RBR" or pos == "RBS" or pos == "JJ"
                                                   or pos == "VB" or pos == "VBD" or pos == "VBG" or pos == "VBN" or pos == "VBP" or pos == "VBZ")]
    pos_tag_list.append(pos_tokens)
    return pos_tag_list


# Function preprocesses text to lower case and removes standard and custom stop words
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
        # tokens = pos_text(tokens)
        tokens_list.append(pos_text(pos_tag(tokens)))
    return tokens_list


# Function prints list passed to it
def print_list(tokens_list):
    for i in range(len(tokens_list)):
        print(tokens_list[0])


# traverse all the documents and compute a vocabulary corpus containing unique words from all the documents
def compute_vocab_list(tokens_list):
    vocab_set = set()
    counter = 0
    for list in tokens_list:
        for sublist in list:
            for doc in sublist:
                counter += 1
                vocab_set.add(doc)
    # print(counter)
    return vocab_set


# Function to compute frequency of term in a document
def freq(term, document):
    return document.count(term)


# Function to compute term frequency of terms in each document
def compute_tf(processed_text):
    document = []
    counter = 0
    tf_doc = compute_vocab_list(processed_text)

    for i in processed_text:
        for doc in i:
            # print(doc)
            tf_vector = [freq(word, doc) for word in tf_doc]
            counter += 1
            document.append(tf_vector)

    # print(len(document[999]))
    return document


# Function for computing l2 normalization
def l2_normalizer(vect):
    denominator = np.sum([element ** 2 for element in vect])
    return [(element / math.sqrt(denominator)) for element in vect]


# Function for computing normalized document matrix
def compute_document_matrix(document):
    doc_term_matrix_l2_norm = []
    for vect in document:
        doc_term_matrix_l2_norm.append(l2_normalizer(vect))
    # print(np.matrix(doc_term_matrix_l2_norm).shape)
    return doc_term_matrix_l2_norm


# Function for counting number of docs having the word
def number_of_docs_with_word(word, document_list):
    document_count = 0
    for doc in document_list:
        if freq(word, doc) > 0:
            document_count += 1
    return document_count


# Function to compute the idf value for each word in the document list
def compute_idf_value(word, document_list):
    number_of_documents = len(document_list)
    df = number_of_docs_with_word(word, document_list)
    return np.log(number_of_documents / 1 + df)


# Function to compute idf matrix
def compute_idf_matrix(idf_vector):
    idf_matrix = np.zeros((len(idf_vector), len(idf_vector)))  # Fill in zeroes where there is no value
    np.fill_diagonal(idf_matrix, idf_vector)  # Fill the values diagonally
    return idf_matrix


# Function to calculate the dot product of normalized tf and idf matrix
def compute_tf_idf(normalized_doc_matrix, idf_matrix):
    doc_term_matrix_tfidf = []
    for tf_vector in normalized_doc_matrix:
        doc_term_matrix_tfidf.append(np.dot(tf_vector, idf_matrix))

    doc_term_matrix_tfidf_l2 = []
    for tf_vector in doc_term_matrix_tfidf:
        doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
    return np.matrix(doc_term_matrix_tfidf_l2)


def cosine_sim(doc_term_matrix_tfidf_l2, centroids):
    # test_doc_term_matrix = np.asarray(doc_term_matrix_tfidf_l2)
    # test_centroids = np.asarray(centroids)
    prod = np.dot(doc_term_matrix_tfidf_l2, centroids)
    len1 = math.sqrt(np.dot(doc_term_matrix_tfidf_l2, doc_term_matrix_tfidf_l2))
    len2 = math.sqrt(np.dot(centroids, centroids))
    return 1 - (prod / (len1 * len2))


def scatter_plot(df):
    feat1 = df['id'].values
    feat2 = df['details'].values
    X = np.array(list(zip(feat1, feat2)))
    plt.scatter(feat1, feat2, c='black', s=7)
    plt.show()


def k_means(X, K):
    nrow = X.shape[0]
    ncol = X.shape[1]

    # pick K random data points as initial centroids
    initial_centroids = np.random.choice(nrow, K, replace=False)
    centroids = X[initial_centroids]  # actual centroid

    # print(X.shape)
    # print(centroids.shape)

    centroids_old = np.zeros((K, ncol))
    cluster_assignments = np.zeros(nrow)

    while (centroids_old != centroids).any():
        centroids_old = centroids.copy()

        # compute distances between data points and centroids
        dist_matrix = distance_matrix(X, centroids, p=2)
        # print(dist_matrix[0])

        dist = []
        for i in range(nrow):
            column = []
            for j in range(centroids.shape[0]):
                column.append(
                    cosine_sim((list(np.array(X[i]).reshape(-1, ))), (list(np.array(centroids[j]).reshape(-1, )))))
            dist.append(column)
        # cosine_sim((list(np.array(X[0]).reshape(-1,))), (list(np.array(centroids[0]).reshape(-1,))))

        # dist_matrix = cosine_sim(X, centroids)
        # print(type(dist_matrix))
        # print(dist_matrix.shape)

        # step 1: find closest centroid for each data point

        for i in np.arange(nrow):
            # d = dist_matrix[i]
            d = dist[i]
            closest_centroid = (np.where(d == np.min(d)))[0][0]  # implement where
            # associate data point with closest centroid
            cluster_assignments[i] = closest_centroid

        # step 2: recompute centroids
        for k in np.arange(K):
            Xk = X[cluster_assignments == k]
            # print(Xk.shape, k)
            centroids[k] = np.apply_along_axis(np.mean, axis=0, arr=Xk)

    sse = []
    for k in range(K):
        cluster = np.where(cluster_assignments == float(k))[0]
        for element in cluster:
            distance = cosine_sim(list(np.array(centroids[k]).reshape(-1, )),
                                   list(np.array(X[element]).reshape(-1, )))
            sse.append(distance ** 2)
    print(K, np.sum(sse))

    return (centroids, cluster_assignments)


def apply_K_means(data, K):
    k_means_result = k_means(data, K)
    centroids = k_means_result[0]
    cluster_assignments = (k_means_result[1]).tolist()
    return centroids, cluster_assignments


def print_clusters(data, cluster_assignments):
    colors = ['r', 'g', 'b']
    f = lambda x: colors[int(x)]
    cluster_assignments = list(map(f, cluster_assignments))

    my_dpi = 96
    plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('K-means clustering\n', fontsize=12)

    plt.scatter([data[:, 0]], [data[:, 1]], color=cluster_assignments)
    plt.show()


def map_ids_tfidf(cluster_assignments, id, k):
    print("kmeans")
    cluster = []
    for i in range(k):
        cluster.append(np.where(cluster_assignments == float(i))[0])
        print(np.where(cluster_assignments == '0.0')[0])
        print(len(np.where(cluster_assignments == '1.0')[0]))
        print(len(np.where(cluster_assignments == '2.0')[0]))



def main():
    path = input("Please enter the file path: ")
    # df = read_file(path)
    df, id = read_csv_file(path)

    processed_text = pre_process(df)
    # processed_text.pop(len(processed_text)-1) #to remove the last empty row

    vocabulary = compute_vocab_list(processed_text)
    document = compute_tf(processed_text)  # list of size 1000 with 1/0 comma-seperated for each word

    normalized_doc_matrix = compute_document_matrix(document)

    # For every word in the vocabulary count documents word is present in and compute word's idf value
    idf_vector = [compute_idf_value(word, processed_text) for word in vocabulary]
    idf_matrix = compute_idf_matrix(idf_vector)

    # print(idf_vector)
    # print(idf_matrix.shape)

    doc_term_matrix_tfidf_l2 = compute_tf_idf(normalized_doc_matrix, idf_matrix)
    # for i in range(2,11):
    #     print("K-means for K = ",i)

    for i in range(37,51):
        centroids, cluster_assignments = apply_K_means(doc_term_matrix_tfidf_l2,i)
        #print(type(cluster_assignments[0]))

    # print_clusters(centroids, cluster_assignments)
    # map_ids_tfidf(cluster_assignments, id, 2)


        cluster = []
        for j in range(i):
            #print(float(j))
            cluster.append(np.argwhere(np.array(cluster_assignments) == float(j)))

        #print("k=", i)
        for i in range(len(cluster)):
            cluster[i] = [item for sublist in cluster[i] for item in sublist]
            #print(len(list(map(id.__getitem__, cluster[i]))), list(map(id.__getitem__, cluster[i])))

    # print(len(np.argwhere(np.array(cluster_assignments) == 0.0)))
    # print(len(np.argwhere(np.array(cluster_assignments) == 1.0)))

if __name__ == '__main__':
    main()