import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gather data
df = pd.read_csv('product_data.csv')

# Preprocess the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['product_description'])

# Train the model
model = cosine_similarity(X)

# Define a function to search for products


def search_products(query):
    query_vector = vectorizer.transform([query])
    scores = model.dot(query_vector.T).toarray()
    top_results = df[['product_id', 'product_name']].iloc[scores.argsort()[
        0][-5:]]
    return top_results
