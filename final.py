from pyspark.sql import SparkSession
import pandas as pd
import json
import sys

spark = SparkSession.builder.appName("Steam Game Merge").getOrCreate()

with open(sys.argv[1], "r") as f:
    raw_data = json.load(f)

for key in raw_data.keys():
    del raw_data[key]["developer"]
    del raw_data[key]["publisher"]
    del raw_data[key]["userscore"]
    del raw_data[key]["owners"]
    del raw_data[key]["average_2weeks"]
    del raw_data[key]["median_2weeks"]
    del raw_data[key]["score_rank"]
    del raw_data[key]["price"]
    del raw_data[key]["discount"]
    del raw_data[key]["ccu"]
    del raw_data[key]["tags"]
    del raw_data[key]["appid"]

raw_df = pd.DataFrame(
    raw_data["570"],
    index=[
        "570",
    ],
)

for n, key in enumerate(raw_data.keys()):
    if n == 0:
        continue
    tmp_df = pd.DataFrame(
        raw_data[key],
        index=[
            key,
        ],
    )
    raw_df = pd.concat([raw_df, tmp_df])

spark_df = spark.createDataFrame(raw_df)
