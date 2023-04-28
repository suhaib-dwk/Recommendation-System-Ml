import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def hybrid_recommendation(user_id, store_id):
    # Load data
    user_store_ratings = pd.read_csv('user_store_ratings.csv')
    item_data = pd.read_csv('item_data.csv')
    demographic_data = pd.read_csv('demographic_data.csv')

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
        user_id, item_data['item_id'], r_ui=4, verbose=True)
    user_store_ratings = pd.DataFrame(user_store_ratings, columns=[
                                      'user_id', 'item_id', 'rating', 'est'])
    user_store_ratings = user_store_ratings.sort_values(
        by='est', ascending=False)

    # Content-based filtering using LSI
    item_data = item_data.drop(['item_id'], axis=1)
    svd = TruncatedSVD(n_components=100)
    svd_matrix = svd.fit_transform(item_data)
    item_data = pd.DataFrame(svd_matrix, columns=range(100))
    item_data['item_id'] = item_data.index
    content_based_recommendations = pd.DataFrame(columns=['item_id', 'score'])
    for i in range(len(item_data)):
        score = cosine_similarity(
            item_data.iloc[i, :-1].values.reshape(1, -1), item_data.iloc[i, :-1].values.reshape(1, -1))
        content_based_recommendations = content_based_recommendations.append(
            {'item_id': item_data.iloc[i, -1], 'score': score[0][0]}, ignore_index=True)
    content_based_recommendations = content_based_recommendations.sort_values(
        by='score', ascending=False)

    # Demographic filtering using Random Forest
    X = demographic_data.drop(['rating'], axis=1)
    y = demographic_data['rating']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    demographic_predictions = clf.predict(X_test)
    demographic_data = pd.DataFrame(
        {'user_id': X_test['user_id'], 'item_id': X_test['item_id'], 'rating': demographic_predictions})

    # Combine recommendations and assign weights
    recommendations = pd.concat([user_store_ratings[['item_id', 'est']],
                                content_based_recommendations, demographic_data[['item_id', 'rating']]])
    recommendations.rename(columns={
                           'est': 'user_based', 'score': 'content_based', 'rating': 'demographic_based'}, inplace=True)
    recommendations['final_score'] = recommendations['user_based'] * 0.3 + \
        recommendations['content_based'] * 0.4 + \
        recommendations['demographic_based'] * 0.3
    recommendations = recommendations.sort_values(
        by='final_score', ascending=False)
    final_recommendations = recommendations.drop_duplicates(subset='item_id')

    return final_recommendations
