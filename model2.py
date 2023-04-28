import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD


database = pd.read_csv('./ratings_Beauty.csv')
df = database.dropna()

df_process = df.pivot_table(
    values='Rating', index='UserId', columns='ProductId', fill_value=0)


def model


X = df_process.T
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape
product_names = list(X.index)
product_ID = product_names.index(i)
product_ID
correlation_product_ID = correlation_matrix[product_ID]

Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i)

Recommend[0:100]
