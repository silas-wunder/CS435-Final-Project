import pandas as pd
import os
import sys
from tqdm import tqdm

def initial_drop(full_data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
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
        "ccu"]
    return full_data.drop(columns_to_drop, axis=1)

def num_languages(data):
    split = data.split(",")
    return len(split)

def languages(full_data: pd.DataFrame):
    full_data["Num_Languages"] = full_data["languages"].astype(str).apply(num_languages)

def make_timeseries_df(time_dict):
    timeseries_df = pd.DataFrame(data={"date": time_dict.keys(), "count": time_dict.values()})
    timeseries_df["date"] = pd.to_datetime(timeseries_df["date"], unit="ms")
    return timeseries_df

def add_mean_concurrent(full_data: pd.DataFrame, time_series: str) -> pd.DataFrame:
    mean_dict = {}
    for file in tqdm(os.listdir(time_series), desc="Adding Concurrent"):
        game_id = file.split("_")[0]
        local_series = pd.read_csv(os.path.join(time_series, file))
        mean_dict[game_id] = local_series["count"].mean()
    mean_df = pd.DataFrame(data=mean_dict, index=mean_dict.keys(), columns=["Mean Concurrent"])
    return pd.concat([full_data, mean_df], axis=1)

def feature_engineering(full_data: pd.DataFrame):
    full_data["Positive Ratio"] = full_data["positive"] / (full_data["positive"] + full_data["negative"])
    full_data["Negative Ratio"] = full_data["negative"] / (full_data["positive"] + full_data["negative"])

def make_final_data(steam_path: str, time_series_path: str):
    raw_data = pd.read_json(os.path.join(steam_path, "steamspy", "detailed", "steam_spy_detailed.json")).T
    full_data = initial_drop(raw_data)
    languages(full_data)
    full_data = add_mean_concurrent(full_data, time_series_path)
    print(full_data)
    full_data.to_parquet(os.path.join(time_series_path, "finished_data.parquet"))

if __name__ == "__main__":
    make_final_data(sys.argv[1], sys.argv[2])