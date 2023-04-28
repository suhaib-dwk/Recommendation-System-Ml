import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load the data into a pandas DataFrame
df = pd.read_csv("./Prouduct.csv")

# Create a list of strings to represent the products
data = []
for i in range(df.shape[0]):
    type_ = df.iloc[i]["type"]
    color = df.iloc[i]["color"]
    specs = df.iloc[i]["specifications"]
    brand = df.iloc[i]["brand"]
    ad = "advertised" if df.iloc[i]["paid_ad"] else "not advertised"
    discount = "discounted" if df.iloc[i]["discount"] else "not discounted"
    data.append(f"{type_} {color} {specs} {brand} {ad} {discount}")

X = df
y = df
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Vectorize the data using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)

# Fit a NearestNeighbors model to the data
model = NearestNeighbors(n_neighbors=10)
model.fit(X)

# Define a function to recommend products


def recommend_products(type, color, specs, brand, ad, discount, location):
    # Preprocess the input data
    input_data = f"{type} {color} {specs} {brand} {ad} {discount}"
    input_vector = vectorizer.transform([input_data])

    # Find the nearest neighbors to the input data
    distances, indices = model.kneighbors(input_vector)

    # Check if any of the nearest neighbors are in the same location as the customer
    df_filtered = df[(df['location'] == location) &
                     (df['discount'] == discount)]
    nearest_neighbors = []

    for index in df_filtered.index:
        nearest_neighbors.append(index)
    # If no neighbors are in the same location, return the nearest neighbors regardless of location
    if len(nearest_neighbors) == 0:
        nearest_neighbors = indices[0]

    # Return the recommended products
    recommendations = []
    for index in nearest_neighbors:
        recommendations.append(df.iloc[index]["productId"])
    return recommendations


# Test the recommendation function
recommendations = recommend_products(
    "shirt", "red", "cotton", "nike", "advertised", "discounted", "jenen")
print(recommendations)

score = model.score(recommendations)
print(f'Model accuracy: {score:.2f}')
