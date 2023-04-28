import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_recommendation_system(data_path, user_preferences, n):
    # Load data from csv file
    df = pd.read_csv(data_path)

    # Filter data for user preferences
    df = df[df['color'] == user_preferences['color']]
    df = df[df['type'] == user_preferences['type']]
    df = df[df['product_size'] == user_preferences['size']]
    df = df[df['location_store'] == user_preferences['location']]
    df = df[df['ad_price'] <= user_preferences['price']]

    # Define item-based collaborative filtering model
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)
    item_collab_model = KNNWithMeans(
        k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    item_collab_model.fit(dataset.build_full_trainset())

    # Define content-based filtering model
    tfidf = TfidfVectorizer(stop_words='english')
    content_based_model = tfidf.fit_transform(
        df[['type', 'color', 'price', 'discount', 'brand', 'name_store', 'specifications', 'location_store', 'product_size']])

    # Create empty list to store recommendations
    recs = []
    # Iterate through each item in the dataset
    for item in df.itertuples():
        # Get the predicted rating from each model
        item_collab_rec = item_collab_model.predict(item[1], item[2]).est

        item_content_vec = tfidf.transform(df[df['itemId'] == item[2]][[
                                           'type', 'color', 'price', 'discount', 'brand', 'name_store', 'specifications', 'location_store', 'product_size']])

        content_rec = cosine_similarity(item_content_vec, content_based_model)

        # Create a dictionary for the current item
        item_recs = {
            'itemId': item[2], 'item_collab_rec': item_collab_rec, 'content_rec': content_rec}

        # Use the weightings to calculate the overall recommendation score
        overall_rec = (0.5 * item_collab_rec) + \
            (0.5 * content_rec)

        # Append the overall recommendation score to the dictionary
        item_recs['overall_rec'] = overall_rec

        # Append the dictionary to the recommendations list
        recs.append(item_recs)

    # Sort the recommendations list by the overall recommendation score
    recs = sorted(recs, key=lambda x: x['overall_rec'], reverse=True)

    # Return the top n recommendations
    return recs[:n]


# top_recs = hybrid_recommendation_system(data_path, user_preferences,n);
# print(top_recs)
