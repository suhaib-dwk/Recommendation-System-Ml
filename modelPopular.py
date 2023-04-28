
import pandas as pd


database = pd.read_csv('./ratings_Beauty.csv')


def firstTimeUser(data):
    popular_products = pd.DataFrame(
        data.groupby('ProductId')['Rating'].count())
    most_popular = popular_products.sort_values('Rating', ascending=False)
    json_object = most_popular[0:100].to_json()
    return json_object


print(firstTimeUser(database))
