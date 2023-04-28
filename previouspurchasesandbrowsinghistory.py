import pyrebase
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Gather data
df = pd.read_csv('customer_data.csv')

# Preprocess the data
X = df.drop(['customer_id', 'purchased_item'], axis=1)
y = df['purchased_item']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)


def get_customer_history(customer_id):
    # Initialize Firebase
    config = {
        "apiKey": "your-api-key",
        "authDomain": "your-auth-domain.firebaseapp.com",
        "databaseURL": "https://your-database-url.firebaseio.com",
        "projectId": "your-project-id",
        "storageBucket": "your-storage-bucket.appspot.com",
        "messagingSenderId": "your-messaging-sender-id"
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    # Retrieve the customer's history from the database
    history = db.child("customer_history").child(customer_id).get()

    return history

# Make recommendations


def make_recommendations(customer_id):
    customer_history = get_customer_history(customer_id)
    recommendations = model.predict(customer_history)
    return recommendations


# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model accuracy: {score:.2f}')
