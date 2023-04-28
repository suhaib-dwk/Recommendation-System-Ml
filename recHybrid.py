import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy, KNNWithMeans
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# Load the data
# Matrix of item characteristics (n_items, n_features)
items = np.load('items.npy')
# Matrix of user preferences (n_users, n_features)
preferences = np.load('preferences.npy')
# Matrix of previous purchases (n_users, n_items)
purchases = np.load('purchases.npy')
ratings = np.load('ratings.npy')  # Matrix of ratings (n_users, n_items)
# Vector of item popularity scores (n_items,)
popularity = np.load('popularity.npy')
location = np.load('location.npy')  # Vector of user locations (n_users,)
age = np.load('age.npy')  # Vector of user ages (n_users,)
# Matrix of item proximities (n_items, n_proximities)
proximity = np.load('proximity.npy')
# Vector of item advertisement scores (n_items,)
advertisement = np.load('advertisement.npy')
offers = np.load('offers.npy')  # Vector of item offer scores (n_items,)

# Combine the relevant data into a single feature matrix for content-based filtering
X_content = np.concatenate((items, preferences, purchases, ratings, popularity[:, np.newaxis], location[
                           :, np.newaxis], age[:, np.newaxis], proximity, advertisement[:, np.newaxis], offers[:, np.newaxis]), axis=1)

# Use a TfidfVectorizer to convert the ratings matrix into a document-term matrix for collaborative filtering
vectorizer = TfidfVectorizer()
X_collab = vectorizer.fit_transform(ratings)

# Use Surprise to train a collaborative filtering model using the ratings data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(pd.DataFrame(ratings), reader)
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)
model = SVD()
model.fit(train_set)
predictions = model.test(test_set)
accuracy.rmse(predictions)

# Define a function that recommends items for a given user using all three types of filtering


def recommend(user_id):
    # Use the SVC classifier to predict the ratings that the user would give to each item based on the ratings that the user would give to each item based on the content-based features
    classifier = svm.SVC()
    classifier.fit(X_content, ratings[:, user_id])
    ratings_content = classifier.predict(X_content)

    # Use the collaborative filtering model to predict the ratings that the user would give to each item
    ratings_collab = model.predict(user_id, np.arange(items.shape[0]))

    # Use KNNWithMeans to predict the ratings that the user would give to each item based on demographic features
    knn = KNNWithMeans(k=5, sim_options={
                       'name': 'cosine', 'user_based': False})
    knn.fit(X_demographic)
    ratings_demographic = knn.predict(user_id, np.arange(items.shape[0]))

    # Combine the ratings from all three types of filtering
    ratings_combined = (ratings_content + ratings_collab +
                        ratings_demographic) / 3

    # Recommend the top-rated items
    recommendations = np.argsort(-ratings_combined)
    return recommendations


# Test the recommendation function
recommendations = recommend(0)
print(recommendations)
