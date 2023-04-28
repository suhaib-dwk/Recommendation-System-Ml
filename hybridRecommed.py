import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_recommendation_system(data_path, user_preferences):
    # Load data from csv file
    df = pd.read_csv(data_path)

    # Filter data for best payment and discount
    df = df[df['ad_price'] == df.loc[df['ad_price'] > 0]['ad_price'].max()]
    df = df[df['discount'] == df.loc[df['discount'] > 0]['discount'].min()]

    # Define item-based collaborative filtering model
    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)

    item_collab_model = KNNWithMeans(
        k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

    item_collab_model.fit(dataset.build_full_trainset())

    user_item_matrix = dataframe.pivot(
        index='user_id', columns='productId', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    print(user_item_matrix)

    # Define content-based filtering model
    tfidf = TfidfVectorizer(stop_words='english')

    content_based_model = tfidf.fit_transform(
        df[['type', 'color', 'ad_price', 'discount', 'brand', 'name_store', 'specifications', 'location_store', 'product_size']])

    # Define demographic-based filtering model
    demographic_based_model = df[(df['location'] == user_preferences['location']) &
                                 (df['color'] == user_preferences['color']) &
                                 (df['size'] == user_preferences['size']) &
                                 (df['age'] == user_preferences['age']) &
                                 (df['gender'] == user_preferences['gender'])]

    # Create empty list to store recommendations
    recs = []
    # Iterate through each item in the dataset
    for item in df.itertuples():

        # Get the predicted rating from each model
        item_collab_rec = item_collab_model.predict(item[1], item[2]).est

        # Here is what you should do with content_based_model,
        item_content_vec = tfidf.transform(df[df['itemId'] == item[2]][[
                                           'type', 'color', 'ad_price', 'discount', 'brand', 'name_store', 'specifications', 'location_store', 'product_size']])

        content_rec = cosine_similarity(item_content_vec, content_based_model)

        demographic_rec = demographic_based_model[demographic_based_model['itemId']
                                                  == item[2]]['rating'].values[0]

        # Create a dictionary for the current item
        item_recs = {'itemId': item[2], 'item_collab_rec': item_collab_rec,
                     'content_rec': content_rec, 'demographic_rec': demographic_rec}

        # Use the weightings to calculate the overall recommendation score
        overall_rec = (0.3 * item_collab_rec) + (
            0.4 * content_rec) + (0.3 * demographic_rec)

        # Append the overall recommendation score to the dictionary
        item_recs['overall_rec'] = overall_rec

        # Append the dictionary to the recommendations list
        recs.append(item_recs)

    # Sort the recommendations list by the overall recommendation score
    recs = sorted(recs, key=lambda x: x['overall_rec'], reverse=True)

    # Return the top n recommendations
    return recs
