"""
Author: Dibyanshu Chatterjee
Username: dc7017@rit.edu
"""

import os
import chardet
import pandas as pd
from nltk.corpus import stopwords
from tabulate import tabulate
import classification as cl
import clustering as cus
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')


def load_input_data(filename):
    """
    Aids in loading and cleaning the input data
    :param filename: The String filename to load
    :return: Pandas df
    """
    column_names = ['id', 'title', 'summary', 'rating', 'name', 'label', 'author', 'description', 'type',
                    'downloads',
                    'useCount',
                    'sampleUrl', 'downloadUrl', 'dateModified', 'remoteFeed', 'numComments', 'commentsUrl', 'tags',
                    'category',
                    'protocols', 'serviceEndpoint', 'version', 'wsdl', 'data formats', 'apigroups', 'example',
                    'clientInstall',
                    'authentication', 'ssl', 'readonly', 'VendorApiKits', 'CommunityApiKits', 'blog', 'forum',
                    'support',
                    'accountReq', 'commercial', 'provider', 'managedBy', 'nonCommercial', 'dataLicensing', 'fees',
                    'limits',
                    'terms', 'company', 'updated']

    file_path = os.getcwd() + "/data/" + filename

    # Open file in binary mode
    with open(file_path, 'rb') as f:
        # Read a chunk of data to analyze the encoding
        rawdata = f.read(10000)
        result = chardet.detect(rawdata)

    # Read the file into a list of dictionaries
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        data = [
            dict(zip(column_names, line.strip().split('$#$')))
            for line in f
        ]

    # Load the data into a pandas DataFrame
    df = pd.DataFrame(data)
    return df


def preprocess_data(data):
    """
    Aids in preprocessing and balancing the training data
    :param data: The input data - Pandas df
    :return: Pandas df of preprocessed data
    """
    # drop missing values
    data.dropna(inplace=True)
    # Select categories with a good number of services to avoid imbalance issue
    categories = data['category'].value_counts().index[data['category'].value_counts() > 50]
    data = data[data['category'].isin(categories)]

    # Remove stop words and punctuation from description column
    stop_words = set(stopwords.words('english'))
    columns_to_process = ['description']

    for col in columns_to_process:
        data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
        data[col] = data[col].str.replace('[^\w\s]', '')

    return data


def perform_feature_selection(data):
    """
    Perform feature selections on preprocessed data
    :param data: Pandas dataframe with preprocessed data
    :return: Numpy array of selected feature data
    """
    # Select columns to use for feature selection

    columns = list(data.columns)
    if 'category' in columns:
        columns.remove('category')
    columns = pd.Index(columns)

    # Set the random seed
    np.random.seed(123)

    # Combine selected columns into a new corpus of text data
    corpus = data[columns].apply(lambda x: ' '.join(x), axis=1)

    # Vectorize text data using tf-idf
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(corpus)

    # Topic modeling using LDA
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    X_lda = lda.fit_transform(X_tfidf)

    # Word embedding modelling using Word2Vec
    sentences = [doc.split() for doc in corpus]
    model = Word2Vec(sentences, min_count=1, vector_size=100)
    X_w2v = np.array([np.mean([model.wv.get_vector(word) for word in words], axis=0) for words in sentences])
    X_w2v[X_w2v < 0] = 0
    return X_tfidf, X_lda, X_w2v


def start_clustering(data):
    """
    Aids in performing the clustering on input data
    :param data: Pandas df with preprocessed ata
    :return: None
    """
    # Exclude the category of a service from the attribute list
    data = data.drop(['category'], axis=1)
    feature_models = perform_feature_selection(data=data)
    count = 0
    for feature_model in feature_models:
        count += 1
        cus.perform_clustering_and_evaluate(feature_model=feature_model, feature_num=count)


def main():
    """
    The driver code
    :return: None
    """
    start_time = time.time()
    input_data = load_input_data("api.txt")

    # preprocess the data
    data = preprocess_data(data=input_data)

    print(data.columns)

    # perform feature selection
    models = perform_feature_selection(data=data)
    counter = 0

    # Train models for classification
    for model in models:
        counter += 1
        print("Model Approach ", counter)
        cl.model_learning(data=data, model=model, counter=counter)
        print()

    # Predict categories in test suite by leveraging classification.py
    predicted_data = cl.predict_categories(new_data=data.drop('category', axis=1))

    print(tabulate(predicted_data.head(5), headers='keys', tablefmt='psql'))

    # Perform clustering with leveraging clustering.py code
    start_clustering(data=data)

    end_time = time.time()

    total_time = end_time - start_time

    print(f'Total time taken by the code to run is: {total_time}')


if __name__ == '__main__':
    main()
