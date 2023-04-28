import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import mean_squared_error
from my_module import preprocess_data
from sklearn.neural_network import MLPRegressor

# Load and preprocess the data
data = pd.read_csv("./productData.csv")
data = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[["user_id", "item_id", "type", "color", "price", "brand"]], data["rating"], test_size=0.2)

# Train an advanced collaborative filtering model
collab_model = MLPRegressor()
collab_model.fit(X_train[["user_id", "item_id"]], y_train)

# Train an advanced content-based recommendation model
vectorizer = HashingVectorizer()
content_model = RandomForestRegressor()
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
