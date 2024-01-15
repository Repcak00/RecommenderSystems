import os
import re
from datetime import datetime

import pandas as pd
import unicodedata
from lxml import html
from lxml.etree import ParserError

data_dir = "data"

unprocessed_reviews_path = os.path.join(data_dir, "sweaters_reviews.json")
processed_reviews_path = os.path.join(data_dir, "sweaters_reviews_processed.json")

unprocessed_meta_path = os.path.join(data_dir, "sweaters_meta.json")
processed_meta_path = os.path.join(data_dir, "sweaters_meta_processed.json")

sweaters_ids_path = os.path.join(data_dir, "sweaters_ids.json")


non_character_re = re.compile(r"[^\w\s]")
newline_re = re.compile("\n")
multi_whitespace_re = re.compile(r"\s+")


def strip_html(s):
    return str(html.fromstring(s).text_content())


def parse_description(description):
    ret_desc = list()
    # parse descriptions
    for sent in description:
        if sent != "":
            try:
                stripped_line = strip_html(sent).strip()
                decoded_line = unicodedata.normalize("NFKC", stripped_line)
                ret_desc.append(decoded_line)
            except ParserError:
                continue
    # join lines of description and substitute unwanted characters
    if (concat_desc := " ".join(ret_desc)) and ret_desc:
        concat_desc = non_character_re.sub(" ", concat_desc)
        concat_desc = newline_re.sub(" \n ", concat_desc)
        concat_desc = multi_whitespace_re.sub(" ", concat_desc)
        return concat_desc.strip()
    return None


def parse_title(title):
    if not title:
        return None
    title = non_character_re.sub(" ", title)
    title = newline_re.sub(" \n ", title)
    title = multi_whitespace_re.sub(" ", title)
    if title == "":
        return None
    try:
        title = strip_html(title).strip()
        title = unicodedata.normalize("NFKC", title)
    except ParserError:
        return None
    return title.strip()


if __name__ == "__main__":
    if not os.path.exists(processed_reviews_path):
        print("Processing reviews data ...")
        df_sweaters_reviews = pd.read_json(unprocessed_reviews_path)

        # drop useless columns
        columns_to_drop = ["image", "reviewTime", "style"]
        df_sweaters_reviews.drop(columns=columns_to_drop, inplace=True)

        # convert unix timestamps column to datetime values
        df_sweaters_reviews["unixReviewTime"] = df_sweaters_reviews[
            "unixReviewTime"
        ].apply(lambda x: datetime.fromtimestamp(x))

        # replace NaNs with 0s in vote
        df_sweaters_reviews["vote"] = df_sweaters_reviews["vote"].fillna(value=0)

        # replace float with int in vote
        df_sweaters_reviews["vote"] = df_sweaters_reviews["vote"].astype(int)

        # rename asin to item_id
        df_sweaters_reviews.rename(columns={"asin": "item_id"}, inplace=True)

        # rename reviewerID to reviewer_id
        df_sweaters_reviews.rename(columns={"reviewerID": "reviewer_id"}, inplace=True)

        # drop null values
        df_sweaters_reviews.dropna(inplace=True)

        # drop duplicates
        df_sweaters_reviews.drop_duplicates(inplace=True)

        # reset index
        df_sweaters_reviews.reset_index(inplace=True, drop=True)

        # save sweaters reviews
        print("Saving processed reviews data ...")
        df_sweaters_reviews.to_json(
            processed_reviews_path, orient="records", indent=4, force_ascii=False
        )

        # deallocate space in memory
        del df_sweaters_reviews
    else:
        print(
            f"{processed_reviews_path} already exists. Skipping processing sweaters reviews"
        )

    if not os.path.exists(processed_meta_path):
        print("Processing metadata ...")
        df_sweaters_meta = pd.read_json(unprocessed_meta_path)
        df_sweaters_ids = pd.read_json(sweaters_ids_path)
        sweaters_ids = list(df_sweaters_ids[0])

        # drop useless columns
        columns_to_drop = [
            "date",
            "feature",
            "rank",
            "imageURL",
            "imageURLHighRes",
            "fit",
            "main_cat",
            "tech1",
            "tech2",
            "details",
            "similar_item",
        ]
        df_sweaters_meta.drop(columns=columns_to_drop, inplace=True)

        # extract relevant category (sex based e.g. Men/Women)
        df_sweaters_meta["subcategory"] = df_sweaters_meta["category"].apply(
            lambda x: x[1]
        )
        df_sweaters_meta.drop(columns=["category"], inplace=True)

        # parse description
        df_sweaters_meta["description"] = df_sweaters_meta["description"].apply(
            lambda x: parse_description(x)
        )

        # parse title
        df_sweaters_meta["title"] = df_sweaters_meta["title"].apply(
            lambda x: parse_title(x)
        )

        # rename asin column to item_id
        df_sweaters_meta.rename(columns={"asin": "item_id"}, inplace=True)

        # in also_view and also_buy remove non-sweater items
        for column in ("also_view", "also_buy"):
            df_sweaters_meta[column] = df_sweaters_meta[column].apply(
                lambda x: [item for item in x if item in sweaters_ids] if x else []
            )

        # parse price (if multiple values, take minimum)
        price_regex = re.compile(r"\d+\.\d\d")
        df_sweaters_meta["price"] = df_sweaters_meta["price"].apply(
            lambda x: min([float(price) for price in price_regex.findall(x)])
            if x
            else x
        )

        # drop null values
        df_sweaters_meta.dropna(inplace=True)

        # drop duplicates
        df_sweaters_meta.drop_duplicates(subset=["item_id"], inplace=True)

        # reset index
        df_sweaters_meta.reset_index(inplace=True, drop=True)

        # save sweaters meta
        print("Saving processed metadata ...")
        df_sweaters_meta.to_json(
            processed_meta_path, orient="records", indent=4, force_ascii=False
        )

        # deallocate space in memory
        del df_sweaters_meta
    else:
        print(
            f"{processed_meta_path} already exists. Skipping processing sweaters metadata"
        )
