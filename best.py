from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from surprise import Reader, Dataset
from sklearn.tree import DecisionTreeClassifier
from surprise.model_selection import train_test_split
from surprise import SVD

demographic_data = pd.DataFrame({'user_id': [1, 2, 3, 4],
                                 'age': [25, 30, 35, 40],
                                 'gender': ['male', 'female', 'male', 'female'],
                                 'size': ['S', 'M', 'L', 'XL'],
                                 'location': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
                                 'color': ['red', 'blue', 'green', 'black'],
                                 'budget': [1000, 1500, 2000, 2500]
                                 })

# create a dataframe
product_data = pd.DataFrame({'product_id': [1, 2, 3, 4],
                             'type': ['shirt', 'pants', 'shoes', 'jacket'],
                             'color': ['red', 'blue', 'green', 'black'],
                             'price': [50, 75, 100, 150],
                             'discount': [10, 20, 30, 40],
                             'brand': ['Nike', 'Adidas', 'Puma', 'Reebok'],
                             'store_name': ['Store A', 'Store B', 'Store C', 'Store D'],
                             'category': ['men', 'women', 'kids', 'unisex'],
                             'store_location': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
                             'product_sizes': [['S', 'M', 'L'], ['M', 'L', 'XL'], ['8', '9', '10'], ['M', 'L', 'XL']]
                             })

# create a dataframe
user_ratings = pd.DataFrame({'user_id': [1, 2, 3, 4],
                            'product_id': [1, 2, 3, 4],
                             'rating': [4, 3, 5, 2],
                             'timestamp': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01']
                             })


def calculate_user_based_score(user_id, user_ratings):
    # Define the reader
    reader = Reader(rating_scale=(1, 5))

    # Create the dataset
    data = Dataset.load_from_df(
        user_ratings[['user_id', 'product_id', 'rating']], reader)

    # Split the dataset into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Use SVD algorithm
    algo = SVD()

    # Train the model
    algo.fit(trainset)

    # Make a prediction
    prediction = algo.predict(user_id, testset)
    return prediction.est


def calculate_content_based_score(product_id, product_data):
    # Select the product description
    product_description = product_data.loc[product_data['product_id']
                                           == product_id, 'description'].iloc[0]

    # Vectorize the product description
    vectorizer = TfidfVectorizer()
    product_vectors = vectorizer.fit_transform([product_description])

    # Calculate the cosine similarity with other products
    cosine_similarities = cosine_similarity(
        product_vectors, vectorizer.transform(product_data['description'])).flatten()

    # Return the average similarity score
    return cosine_similarities.mean()


def calculate_demographic_score(user_id, demographic_data):
    # select user demographic data
    user_data = demographic_data.loc[demographic_data['user_id'] == user_id]

    # define the classifier
    clf = DecisionTreeClassifier()

    # train the classifier
    clf.fit(demographic_data.drop(
        columns=['user_id']), demographic_data['score'])

    # make a prediction
    score = clf.predict(user_data.drop(columns=['user_id']))
    return score


def hybrid_recommendation(user_id, product_id, user_ratings, product_data, demographic_data):
    # calculate user-based score
    user_based_score = calculate_user_based_score(user_id, user_ratings)

    # calculate content-based score
    content_based_score = calculate_content_based_score(
        product_id, product_data)

    # calculate demographic score
    demographic_score = calculate_demographic_score(user_id, demographic_data)

    recommendation = []
    # calculate final recommendation
    recommendation = (user_based_score * 0.3) + \
        (content_based_score * 0.4) + (demographic_score * 0.3)
    return recommendation


hy = hybrid_recommendation(1, 2, user_ratings, product_data, demographic_data)
print(hy)
