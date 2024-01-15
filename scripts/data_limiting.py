import os

import pandas as pd

data_dir = "data"
processed_reviews_with_sentiment_path = os.path.join(
    data_dir, "sweaters_reviews_sentiment.json"
)

item_min_num_reviews = 10
user_min_num_unique_items = 3

limited_reviews_path = os.path.join(
    data_dir,
    f"limited_{item_min_num_reviews}_{user_min_num_unique_items}_sweaters_reviews_sentiment.json",
)


if __name__ == "__main__":
    if not os.path.exists(limited_reviews_path):
        print(
            f"Limiting sweaters reviews to items with more than {item_min_num_reviews} reviews"
        )
        df_sweaters_reviews = pd.read_json(processed_reviews_with_sentiment_path)

        item_counts = df_sweaters_reviews["item_id"].value_counts()
        df_filtered_sweaters_reviews = df_sweaters_reviews[
            df_sweaters_reviews["item_id"].isin(
                item_counts.index[item_counts > item_min_num_reviews]
            )
        ]

        print(
            f"Limiting sweaters reviews to users with more than {user_min_num_unique_items} unique items"
        )
        users_grouped_by_item = df_filtered_sweaters_reviews.groupby("reviewer_id")[
            "item_id"
        ].nunique()
        filtered_users_grouped_by_item = users_grouped_by_item[
            users_grouped_by_item > user_min_num_unique_items
        ]
        df_filtered_sweaters_reviews = df_filtered_sweaters_reviews[
            df_filtered_sweaters_reviews["reviewer_id"].isin(
                filtered_users_grouped_by_item.index
            )
        ]

        df_filtered_sweaters_reviews.to_json(
            limited_reviews_path, orient="records", indent=4, force_ascii=False
        )
    else:
        print(
            f"{limited_reviews_path} already exists. Skipping limiting reviews dataset"
        )
