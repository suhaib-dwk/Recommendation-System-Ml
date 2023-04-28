import pandas as pd
from surprise import KNNWithMeans
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_recommendation(user_id, store_data, product_id, num_recommendations=100):
    # Load the data from a CSV file
    dataframe = pd.read_csv("./productData.csv")

    # Create a user-item matrix
    user_item_matrix = dataframe.pivot(
        index='user_id', columns='productId', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    print(user_item_matrix)

    # Create a cosine similarity matrix for the product data
    product_data = dataframe[['type', 'color', 'price', 'discount', 'brand', 'paid_ad',
                              'store_name', 'category', 'location_store', 'product_sizes']]

    if product_data['productId'].dtype == 'object':
        product_data = product_data.drop_duplicates(
            subset='productId', keep='first')
        # Set the index as the product Id
        product_data = product_data.set_index('productId')
        # Calculate cosine similarity between products
        cosine_sim = cosine_similarity(product_data)
        cosine_sim = pd.DataFrame(
            cosine_sim, index=product_data.index, columns=product_data.index)

        # Get the user's previous purchase history and searches
        user_history = store_data[store_data.user_id == user_id]
        # Get the products in the user's history
        user_history_products = list(user_history['productId'].unique())
        # Get the top N similar products for a given product
        similar_products = cosine_sim[product_id].sort_values(ascending=False)[
            1:num_recommendations+1]
        # Combine the user's previous purchase history and the similar products
        recommended_products = list(
            set(similar_products).intersection(user_history_products))

    # Create a user-based collaborative filtering model
    algo = KNNWithMeans(k=50, sim_options={'user_based': True})
    algo.fit(user_item_matrix)

    # Get the top N similar products to the input product using cosine similarity
    input_product_index = product_data.index.get_loc(product_id)
    top_n_recs = recommended_products.iloc[input_product_index].sort_values(ascending=False)[
        1:11]

    # Get the items the user has interacted with
    user_inner_id = algo.trainset.to_inner_uid(user_id)
    user_items = set(
        [iid for (uid, iid, _) in algo.trainset.ur[user_inner_id]])

    # Use the predicted ratings and the cosine similarity scores to generate the recommendations
    recs = []
    item_collab_rec = []
    content_rec = []
    for iid in top_n_recs.index:
        if iid in user_items:
            continue
        recs.append(iid)
        item_collab_rec.append(algo.predict(user_id, iid).est)
        content_rec.append(top_n_recs[iid])

    # Assign weights to the collaborative and content-based scores
    collaborative_weight = 0.5
    content_weight = 0.5

    # Combine the predicted ratings with the weights
    recs_df = pd.DataFrame(
        {'product_id': recs, 'collaborative_score': item_collab_rec, 'content_score': content_rec})
    recs_df['weighted_score'] = recs_df['collaborative_score'] * \
        collaborative_weight + recs_df['content_score'] * content_weight

    # Sort the recommendations by the weighted score
    recs_df = recs_df.sort_values(by='weighted_score', ascending=False)
    # Return the top 5 recommendations
    return recs_df.head(5)


hy = hybrid_recommendation(3, 2)
print(hy)
