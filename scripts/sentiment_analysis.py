import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

data_dir = "data"
processed_reviews_path = os.path.join(data_dir, "sweaters_reviews_processed.json")
processed_reviews_with_sentiment_path = os.path.join(
    data_dir, "sweaters_reviews_sentiment.json"
)


def analyze_sentiment(row):
    if row is not None:
        return distilled_student_sentiment_classifier(row)[0]["label"]
    else:
        return np.NaN


if __name__ == "__main__":
    if not os.path.exists(processed_reviews_with_sentiment_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=False,
            device=device,
        )

        df_sweaters_reviews = pd.read_json(processed_reviews_path)

        # clip text to 512 (model max)
        df_sweaters_reviews["clipped_reviewText"] = df_sweaters_reviews[
            "reviewText"
        ].str.slice(stop=512)

        tqdm.pandas(desc="Analyzing Sentiments")
        df_sweaters_reviews["sentiment"] = df_sweaters_reviews[
            "clipped_reviewText"
        ].progress_apply(analyze_sentiment)

        df_sweaters_reviews.drop("clipped_reviewText", axis=1, inplace=True)

        df_sweaters_reviews.to_json(
            processed_reviews_with_sentiment_path,
            orient="records",
            indent=4,
            force_ascii=False,
        )
    else:
        print(
            f"{processed_reviews_with_sentiment_path} already exists. Skipping sentiment analysis"
        )
