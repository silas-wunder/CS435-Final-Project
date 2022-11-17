from pyspark.sql import SparkSession
import pandas as pd
import json
import sys

spark = SparkSession.builder.appName("Steam Game Merge").getOrCreate()
with open("./input/steamspy/detailed/steam_spy_detailed.json", "r") as f:
    # with open('/Final/input/steamspy/detailed/steam_spy_detailed.json', 'r') as f:
    raw_file = json.load(f)


full_data = pd.DataFrame.from_records(raw_file).T
full_data.drop(
    [
        "appid",
        "developer",
        "publisher",
        "score_rank",
        "userscore",
        "owners",
        "average_2weeks",
        "median_2weeks",
        "price",
        "discount",
        "ccu",
    ],
    axis=1,
    inplace=True,
)
full_data.rename(
    {
        "name": "Name",
        "positive": "Positive Reviews",
        "negative": "Negative Reviews",
        "average_forever": "Average Playtime",
        "median_forever": "Median Playtime",
        "initialprice": "Price",
        "languages": "Languages",
        "genre": "Genres",
        "tags": "Tags",
    },
    axis=1,
    inplace=True,
)

spark_df = spark.createDataFrame(full_data)
