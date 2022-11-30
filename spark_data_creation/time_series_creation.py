import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, MapType, LongType
from pyspark.ml.linalg import DenseMatrix
import sys

def make_timeseries_folder(steam_path, output_path):
    context = SparkContext(appName="data_testing").getOrCreate()
    spark = SparkSession(context).builder.getOrCreate()
    time_series_path = os.path.join(steam_path, "steam_charts", "steam_charts.json")
    full_stream = spark.read.json(time_series_path).cache()

    
    '''
    for i in tqdm(range(full_data.shape[0]), desc="Processing"):
        time_dict = full_data.iloc[i][0]
        time_df = make_timeseries_df(time_dict)
        game_id = full_data.iloc[i]["index"]
        time_df.to_csv(os.path.join(output_path, f"{game_id}_counts.csv"))
        '''



if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        make_timeseries_folder(data_path, output_path)
    