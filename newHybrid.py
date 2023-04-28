import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("./productData.csv")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[["user_id", "item_id", "item_description", "color", "price", "brand"]], data["rating"], test_size=0.2)

# Train a collaborative filtering model
collab_model = NearestNeighbors(n_neighbors=10)
collab_model.fit(X_train[["user_id", "item_id"]], y_train)

# Train a content-based model
vectorizer = TfidfVectorizer()
content_model = LinearRegression()
content_model.fit(vectorizer.fit_transform(
    X_train[["item_description", "color", "price", "brand"]]), y_train)


# Define a function to make recommendations using the hybrid recommendation system


def recommend(user_id, item_id, item_description, color, price, brand):
    collab_prediction = collab_model.predict(user_id, item_id)
    content_prediction = content_model.predict(
        vectorizer.transform([item_description, color, price, brand]))
    return (collab_prediction + content_prediction) / 2


# Evaluate the performance of the hybrid recommendation system on the testing data
y_pred = [recommend(user_id, item_id, item_description, color, price, brand) for user_id, item_id, item_description,
          color, price, brand in X_test[["user_id", "item_id", "item_description", "color", "price", "brand"]]]
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


def get_recommendations(user_id, num_recommendations):
    items = data[["item_id", "item_description",
                  "color", "price", "brand"]].drop_duplicates()
    ratings = [recommend(user_id, item_id, item_description, color, price, brand)
               for item_id, item_description, color, price, brand in items.values]
    items["ratings"] = ratings
    return items.sort_values("ratings", ascending=False).head(num_recommendations)


# Get the top 5 recommendations for user with id 1
print(get_recommendations(1, 5))
# Get the top 10 recommendations for user with id 2
print(get_recommendations(2, 10))
