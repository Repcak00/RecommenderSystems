import numpy as np
import torch
from tqdm.autonotebook import tqdm
import pandas as pd
from scripts.collaborative_recommendations import (
    get_recommendation_collaborative,
    measure_distances_indices_collaborative,
)
import os
from PIL import Image

cwd = os.getcwd()
parent_directory = os.path.abspath(os.path.join(cwd, os.pardir))
data_directory = os.path.join(parent_directory, "data")
df_train = pd.read_json(
    os.path.join(data_directory, "limited_10_3_sweaters_reviews_sentiment_train.json"),
    orient="split",
)

distances, indices = measure_distances_indices_collaborative(
    df_train, neighbors_to_scoring=20, metric="cosine"
)


def calculate_mrr(
    df,
    recommendations=None,
    recommendation_function=None,
    n_items=10,
    distances=distances,
    indices=indices,
):
    mrr_values = []
    for user_id, items_bought in df.iterrows():
        if recommendations is None:
            analyzed_recommendations = recommendation_function(
                df_train, user_id, n_items, distances, indices
            )
        else:
            analyzed_recommendations = recommendations.loc[user_id].values[0]

        # Check if any of the analyzed recommendations are in the set of relevant items
        relevant_items = set(items_bought.values[0])
        rank = next(
            (
                i + 1
                for i, item in enumerate(analyzed_recommendations)
                if item in relevant_items
            ),
            n_items + 1,
        )
        # Calculate Reciprocal Rank
        reciprocal_rank = 1 / rank if rank <= n_items else 0

        mrr_values.append(reciprocal_rank)

    # Calculate Mean Reciprocal Rank
    mean_mrr = np.round(np.mean(mrr_values), 3)
    return mean_mrr


def calculate_map(
    df,
    recommendations=None,
    recommendation_function=None,
    n_items=10,
    distances=distances,
    indices=indices,
):
    map_values = []
    for user_id, items_bought in df.iterrows():
        if recommendations is None:
            analyzed_recommendations = recommendation_function(
                df_train, user_id, n_items, distances, indices
            )
        else:
            analyzed_recommendations = recommendations.loc[user_id].values[0]

        # Check if any of the analyzed recommendations are in the set of relevant items
        relevant_items = set(items_bought.values[0])
        precision_at_k = []

        for i, item in enumerate(analyzed_recommendations[:n_items]):
            num_relevant_items = 0
            if item in relevant_items:
                num_relevant_items += 1
            precision_at_k.append(num_relevant_items / len(relevant_items))

        average_precision = np.mean(precision_at_k) if precision_at_k else 0
        map_values.append(average_precision)

    # Calculate Mean Average Precision
    mean_map = np.round(np.mean(map_values), 3)
    return mean_map


def calculate_ndcg(
    df,
    recommendations=None,
    recommendation_function=None,
    n_items=10,
    distances=distances,
    indices=indices,
):
    ndcg_values = []

    for index, row in df.iterrows():
        user_id = index
        items_bought = row
        if recommendations is None:
            analyzed_recommendations = recommendation_function(
                df_train, user_id, n_items, distances=distances, indices=indices
            )
        else:
            analyzed_recommendations = recommendations.loc[user_id].values[0]
        # Calculate DCG
        dcg = 0
        for i, item in enumerate(analyzed_recommendations):
            if item in items_bought.values.any():
                relevance = 1  # Assuming relevance is 1 if the item is bought
            else:
                relevance = 0
            dcg += (2**relevance - 1) / np.log2(
                i + 2
            )  # i+2 because indexing starts from 0

        # Calculate ideal DCG
        ideal_dcg = sum(
            (2**1 - 1) / np.log2(i + 2)
            for i in range(min(n_items, len(items_bought)))
        )

        # Calculate NDCG
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_values.append(ndcg)

    mean_ndcg = np.round(np.mean(ndcg_values), 3)
    return mean_ndcg


def get_table_results(
    df, ranking, recommendation_function, n_items_array=np.arange(2, 21, 2)
):
    results = {}
    for n_items in n_items_array:
        map_value = calculate_map(
            df=df,
            recommendations=ranking,
            recommendation_function=recommendation_function,
            n_items=n_items,
            distances=distances,
            indices=indices,
        )
        mrr_value = calculate_mrr(
            df=df,
            recommendations=ranking,
            recommendation_function=recommendation_function,
            n_items=n_items,
            distances=distances,
            indices=indices,
        )
        ndcg_value = calculate_ndcg(
            df=df,
            recommendations=ranking,
            recommendation_function=recommendation_function,
            n_items=n_items,
            distances=distances,
            indices=indices,
        )
        results[n_items] = {
            "map_value": map_value,
            "mrr_value": mrr_value,
            "ndcg_value": ndcg_value,
        }
    return results


def get_user_images(recommendations, pictures_dir, transform, image_size):
    user_images = []
    for user_idx in tqdm(
        range(len(recommendations)), desc="Images processed", colour="magenta"
    ):
        items_images = {}
        for item_name in recommendations[user_idx]:
            dict_path = os.path.join(pictures_dir, item_name)
            images = os.listdir(dict_path)
            if images:
                image_path = os.path.join(dict_path, images[0])
                image = Image.open(image_path)
            else:
                image = Image.fromarray(
                    np.zeros((image_size, image_size, 3), dtype=np.uint8)
                )
            image = transform(image)
            items_images[item_name] = image
        user_images.append(items_images)
    return user_images


def get_user_item_preds(user_index_array, user_images, users_ids, model, device):
    n_users_preds = {}
    for idx, user_index in tqdm(
        enumerate(user_index_array),
        desc="Predicter processing",
        colour="magenta",
        total=len(user_index_array),
    ):
        predictions = {}
        items_images = user_images[idx]
        for item_name, image in items_images.items():
            pred = (
                model(
                    torch.tensor(user_index).unsqueeze(0).to(device),
                    image.unsqueeze(0).to(device),
                )
                .detach()
                .cpu()
                .item()
            )
            predictions[item_name] = pred
        sorted_items = sorted(
            predictions.items(), reverse=True, key=lambda item: item[1]
        )
        n_users_preds[users_ids[idx]] = sorted_items

    return n_users_preds
