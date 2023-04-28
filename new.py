import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score

# 1. Collect data and create user profiles
df = pd.read_csv("data.csv")

# 2. Preprocess the data
X = df.drop(columns=["item_id", "rating"])
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Preprocessing for demographic-based model
numeric_features = ["age", "income"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

categorical_features = ["gender", "occupation"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)])

# 3. Train a collaborative filtering model
collab_model = NearestNeighbors(n_neighbors=10)
collab_model.fit(X_train[["user_id", "item_id"]], y_train)

# 4. Train a content-based filtering model
vectorizer = TfidfVectorizer()
content_model = LinearRegression()
content_model.fit(vectorizer.fit_transform(
    X_train["item_description"]), y_train)

# 5. Train a demographic-based model
demo_model = LinearRegression()
demo_model.fit(preprocessor.fit_transform(X_train), y_train)

# 6. Combine the predictions from the three models


def recommend(user_id, item_id, item_description):
    collab_prediction = collab_model.predict(user_id, item_id)
    content_prediction = content_model.predict(
        vectorizer.transform([item_description]))
    demo_prediction = demo_model.predict(
        preprocessor.transform(X[X["user_id"] == user_id]))

    return (collab_prediction + content_prediction + demo_prediction) / 3


# 7. Evaluate the performance of the hybrid system
y_pred = [recommend(user_id, item_id, item_description) for (
    user_id, item_id, item_description) in X_test[["user_id", "item_id", "item_description"]].values]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
