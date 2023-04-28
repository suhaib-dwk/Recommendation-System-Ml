import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

demographic_data = pd.DataFrame({'user_id': [1, 2, 3, 4],
                                 'age': [25, 30, 35, 40],
                                 'gender': ['male', 'female', 'male', 'female'],
                                 'size': ['S', 'M', 'L', 'XL'],
                                 'location': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
                                 'color': ['red', 'blue', 'green', 'black'],
                                 'budget': [1000, 1500, 2000, 2500]
                                 })

# create a dataframe
product_data = pd.DataFrame({'item_id': [1, 2, 3, 4],
                             'store_id': [21, 14, 56, 48],
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
                             'store_id': [21, 14, 56, 48],
                            'item_id': [1, 2, 3, 4],
                             'rating': [4, 3, 5, 2],
                             'timestamp': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01']
                             })


def hybrid_recommendation(user_id, store_id, user_ratings, product_data, demographic_data):
    # Load data
    user_store_ratings = user_ratings
    item_data = product_data
    user_preferences = demographic_data

    # User-based collaborative filtering using KNNWithMeans
    user_store_ratings = user_store_ratings[user_store_ratings['store_id'] == store_id]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        user_store_ratings[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    knn = KNNWithMeans(sim_options=sim_options)
    knn.fit(trainset)
    user_store_ratings = knn.predict(
        user_id, item_data['item_id'], r_ui=4, verbose=False)
    user_store_ratings = pd.DataFrame(user_store_ratings, columns=[
                                      'user_id', 'item_id', 'est'])
    print(user_store_ratings)
    # Content-based filtering using LSI
    svd = TruncatedSVD(n_components=100, random_state=42)
    item_data_features = svd.fit_transform(item_data.drop(['item_id'], axis=1))
    item_data_features = pd.DataFrame(item_data_features, columns=[
                                      'svd_feature_' + str(i) for i in range(100)])
    item_data_features['item_id'] = item_data['item_id']
    content_based_recommendations = pd.DataFrame(columns=['item_id', 'score'])
    for i in range(len(item_data_features)):
        score = cosine_similarity(item_data_features.iloc[i, :-1].values.reshape(
            1, -1), item_data_features.iloc[i, :-1].values.reshape(1, -1))
        content_based_recommendations = content_based_recommendations.append(
            {'item_id': item_data_features.iloc[i, -1], 'score': score[0][0]}, ignore_index=True)
    content_based_recommendations = content_based_recommendations.sort_values(
        by='score', ascending=False)

    # Demographic filtering using Random Forest
    X = user_preferences.drop(['user_id'], axis=1)
    y = user_preferences['user_id']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    demographic_filtering_score = rf.score(X_test, y_pred)

    # Combine recommendations
    user_store_ratings['score'] = user_store_ratings['est'] * 0.3
    content_based_recommendations['score'] = content_based_recommendations['score'] * 0.4
    demographic_filtering_score = demographic_filtering_score * 0.3
    combined_recommendations = pd.merge(
        user_store_ratings, content_based_recommendations, on='item_id', how='outer')
    combined_recommendations = combined_recommendations.fillna(0)
    combined_recommendations['score'] = combined_recommendations['score_x'] + \
        combined_recommendations['score_y'] + demographic_filtering_score
    combined_recommendations = combined_recommendations.sort_values(
        by='score', ascending=False)

    return combined_recommendations


print(hybrid_recommendation(2, 14, user_ratings, product_data, demographic_data))
