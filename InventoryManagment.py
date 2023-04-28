import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Gather data
df = pd.read_csv('sales_data.csv')

# Preprocess the data
X = df.drop(['product_id', 'quantity_sold'], axis=1)
y = df['quantity_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Define a function to forecast demand


def forecast_demand(product_data):
    prediction = model.predict([product_data])
    return prediction[0]


# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model accuracy: {score:.2f}')
