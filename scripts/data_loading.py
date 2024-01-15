import json
import os
import pickle

import pandas as pd

data_dir = "data"

items_urls_to_scrape_path = os.path.join(data_dir, "items_urls_to_scrape.pkl")
sweaters_ids_path = os.path.join(data_dir, "sweaters_ids.json")

sweaters_reviews_original_path = os.path.join(
    data_dir, "Clothing_Shoes_and_Jewelry_5.json"
)
sweaters_reviews_path = os.path.join(data_dir, "sweaters_reviews.json")

sweaters_meta_original_path = os.path.join(
    data_dir, "meta_Clothing_Shoes_and_Jewelry.json"
)
sweaters_meta_path = os.path.join(data_dir, "sweaters_meta.json")


if __name__ == "__main__":
    if not os.path.exists(sweaters_ids_path):
        print("Preparing sweaters indices ...")
        # get sweaters only indices
        with open(items_urls_to_scrape_path, "rb") as file:
            items_to_scrape = pickle.load(file)

        sweaters_ids = list(items_to_scrape.keys())

        with open(sweaters_ids_path, "w") as file:
            json.dump(sweaters_ids, file, sort_keys=True, indent=4, ensure_ascii=False)
    else:
        print(
            f"{sweaters_ids_path} already exists. Skipping extracting sweaters indices"
        )

    if not os.path.exists(sweaters_reviews_path):
        print("Preparing sweaters reviews ...")
        # limit reviews to contain only sweaters
        df_reviews = pd.read_json(sweaters_reviews_original_path, lines=True)
        df_reviews_sweaters = df_reviews[df_reviews["asin"].isin(sweaters_ids)]
        df_reviews_sweaters.reset_index(inplace=True, drop=True)
        df_reviews_sweaters.to_json(
            sweaters_reviews_path, orient="records", indent=4, force_ascii=False
        )
    else:
        print(
            f"{sweaters_reviews_path} already exists. Skipping preparing sweaters reviews"
        )

    if not os.path.exists(sweaters_meta_path):
        print("Preparing sweaters meta ...")
        # limit meta to contain only sweaters
        df_meta = pd.read_json(sweaters_meta_original_path, lines=True)
        df_meta_sweaters = df_meta[df_meta["asin"].isin(sweaters_ids)]
        df_meta_sweaters.reset_index(inplace=True, drop=True)
        df_meta_sweaters.to_json(
            sweaters_meta_path, orient="records", indent=4, force_ascii=False
        )
    else:
        print(f"{sweaters_meta_path} already exists. Skipping preparing sweaters meta")
