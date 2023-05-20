import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import nltk
import main

nltk.download('stopwords')


def model_learning(data, model, counter):
    """
    Aids in training different classification models
    :param counter:
    :param data: The Pandas df with preprocessed data
    :param model: The Numpy array with extracted feature model
    :return: None
    """
    # Split data into features and target

    X = model
    y = data['category']

    # Decision tree
    dt = DecisionTreeClassifier()
    dt_scores = cross_val_score(dt, X, y, cv=5)
    print("Decision Tree accuracy:", np.mean(dt_scores))

    # Naive Bayes
    nb = MultinomialNB()
    nb_scores = cross_val_score(nb, X, y, cv=5)
    print("Naive Bayes accuracy:", np.mean(nb_scores))

    # K-nearest neighbors
    knn = KNeighborsClassifier()
    knn_scores = cross_val_score(knn, X, y, cv=5)
    print("KNN accuracy:", np.mean(knn_scores))

    # Create a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Compute the cross-validation scores
    rf_scores = cross_val_score(rf, X, y, cv=5)

    # Print the accuracy
    print("Random forest accuracy:", np.mean(rf_scores))

    # Train the random forest model on the full dataset
    rf.fit(X, y)

    pickle_file = 'random_forest_model_' + str(counter) + '.pkl'

    # Perform pickling for future predictions
    with open(pickle_file, 'wb') as f:
        pickle.dump(rf, f)


def predict_categories(new_data):
    """
    Aids in predicting categories in the future data
    :param new_data: The future preprocessed data
    :return: Pandas df with predicted category values
    """
    # Load the trained model from disk
    with open('random_forest_model_1.pkl', 'rb') as f:
        rf = pickle.load(f)

    # Perform feature selection on new data
    X_tfidf_new, X_lda_new, X_w2v_new = main.perform_feature_selection(new_data)

    # Make predictions using pre-trained model of random forest
    rf_preds = rf.predict(X_tfidf_new)

    # Create a DataFrame with the predictions
    preds_df = pd.DataFrame({'Random Forest': rf_preds})
    preds_df.index = new_data.index

    # Return the DataFrame with the predictions
    return preds_df

