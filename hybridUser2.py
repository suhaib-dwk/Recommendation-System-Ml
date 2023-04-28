import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data into a Pandas dataframe
data = pd.read_csv("clothing_store_data.csv")

# Define the columns to use as features for the content-based filtering
features = ["type", "color", "price", "discount", "brand", "store_name",
            "specifications", "location_store", "proximity", "store_name", "product_sizes"]

# Create a Tf-idf vectorizer and fit it to the features
vectorizer = TfidfVectorizer(use_idf=True)
X = vectorizer.fit_transform(data[features])

# Compute the cosine similarity between the products
similarity = cosine_similarity(X)

# Compute the user-item collaborative filtering based on the ratings
user_item_similarity = 1 - pairwise_distances(data[['user_id', 'item_id', 'rating']].pivot_table(
    index='user_id', columns='item_id').fillna(0), metric='cosine')

# Define the weighting for each recommendation
content_weight = 0.7
user_weight = 0.3

# Combine the similarities with weighting
combined_similarity = content_weight * \
    similarity + user_weight * user_item_similarity

# Make recommendations based on the combined similarity matrix


def recommend_products(product_id, n=5):
    idx = product_id - 1
    scores = list(enumerate(combined_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]
    product_indices = [i[0] for i in scores]
    return data['product_name'].iloc[product_indices]

# Example usage:
# recommend_products(1)
