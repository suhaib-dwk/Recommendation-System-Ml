import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Gather data
df = pd.read_csv('transaction_data.csv')

# Preprocess the data
X = df.drop(['transaction_id', 'fraud'], axis=1)
y = df['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define a function to detect fraud


def detect_fraud(transaction_data):
    prediction = model.predict([transaction_data])
    if prediction[0] == 1:
        return 'Suspicious transaction detected. Please review.'
    else:
        return 'Transaction appears legitimate.'


# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model accuracy: {score:.2f}')
