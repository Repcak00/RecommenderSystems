import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def measure_distances_indices_collaborative(
    user_item_matrix: pd.DataFrame,
    neighbors_to_scoring: int = 5,
    metric: str = "cosine",
):
    model_knn = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=10)
    model_knn.fit(user_item_matrix.values)

    distances, indices = model_knn.kneighbors(
        user_item_matrix.values, n_neighbors=neighbors_to_scoring
    )
    return distances, indices


def get_recommendation_collaborative(
    user_item_matrix,
    user_name="A0604201MBJEJ93VKB77",
    n_items=10,
    distances=None,
    indices=None,
):
    recs = []
    if distances is None and indices is None:
        distances, indices = measure_distances_indices_collaborative(
            user_item_matrix, neighbors_to_scoring=20, metric="cosine"
        )

    try:  # check if user_id is available, if not this is cold start problem
        idx = user_item_matrix.index.get_loc(user_name)
        nearest_users = indices[idx, 1:]  # get nearest users for this user

        scoring = pd.DataFrame(
            data=np.sum(user_item_matrix.iloc[nearest_users], axis=0), columns=["score"]
        ).sort_values(by="score", ascending=False)

        for (
            picked_item
        ) in (
            scoring.index
        ):  # we need to recommend n_items - and check if user listened to it
            if user_item_matrix.loc[user_name, picked_item] != 0.0:
                recs.append(picked_item)
            if len(recs) == n_items:  # break when list is long enough
                break
    except KeyError:
        print("Cold start, recommending 5 most popular items")
        scoring = pd.DataFrame(
            data=np.sum(user_item_matrix, axis=0), columns=["score"]
        ).sort_values(by="score", ascending=False)
        recs = scoring.index[:5]

    return recs


if __name__ == "__main__":
    cwd = os.getcwd()
    # parent_directory = os.path.abspath(os.path.join(cwd, os.pardir))
    data_directory = os.path.join(cwd, "data")
    file_name = "limited_10_3_sweaters_reviews_sentiment.json"

    df_reviews = pd.read_json(
        # os.path.join(data_directory, "reviews_filtered_sentiment.json")
        os.path.join(data_directory, file_name)
    )
    df_collaborative = df_reviews[["reviewer_id", "item_id", "sentiment"]]

    sentiment_mapping = {
        "positive": 2,
        "neutral": 1,
        "negative": -2,
        "nan": 0,
    }  # setting value -1 for situation when user didnt buy item
    df_collaborative.loc[:, "sentiment"] = df_collaborative["sentiment"].replace(
        sentiment_mapping
    )
    df_collaborative = (
        df_collaborative.groupby(["reviewer_id", "item_id"])["sentiment"]
        .min()
        .reset_index()
    )  # removing duplicated reviews by getting min value

    X = df_collaborative[["reviewer_id", "item_id"]].values
    y = df_collaborative["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y, test_size=0.2, random_state=42, stratify=y
    )
    train_df, test_df = train_test_split(
        df_collaborative,
        test_size=0.2,
        random_state=42,
        stratify=df_collaborative["sentiment"],
    )

    df_train = train_df.pivot(
        index="reviewer_id", columns="item_id", values="sentiment"
    ).fillna(-1)

    df_test = test_df.pivot(
        index="reviewer_id", columns="item_id", values="sentiment"
    ).fillna(-1)

    df_train.to_json(
        os.path.join(
            data_directory, "limited_10_3_sweaters_reviews_sentiment_train.json"
        ),
        orient="split",
        indent=4,
        force_ascii=False,
    )
    df_test.to_json(
        os.path.join(
            data_directory, "limited_10_3_sweaters_reviews_sentiment_test.json"
        ),
        orient="split",
        indent=4,
        force_ascii=False,
    )

    recommendations = get_recommendation_collaborative(df_train, "A2UOWHRPP4895I", 10)
    # test = pd.read_json("data/reviews_sentiment_test.json", orient='split')
    print(recommendations)
