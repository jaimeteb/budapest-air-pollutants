import datetime as dt
import glob
import json
import typing as t

import tqdm
import typer

import numpy as np
import pandas as pd

from tensorflow.keras import models


models_dir = "./budapest_air_pollutants/model"
data_dir = "./budapest_air_pollutants/dataset_air_quality"
final_data_dir = "./budapest_air_pollutants/final_data"
ts_cols = ["pm25", "pm10", "o3", "no2", "so2", "co"]
timesteps = 7
INPUT_LEN = 6
OUTPUT_LEN = 6


def load_data() -> t.Dict[str, pd.DataFrame]:
    typer.echo("Loading data")

    # load data from csv files
    dfs = {
        path.split("/")[-1].replace(".csv", ""): pd.read_csv(path)
        for path in glob.glob(f"{data_dir}/*")
    }
    
    # data columns

    # formatting of data
    for name, df in tqdm.tqdm(dfs.items()):
        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        
        for val in ts_cols:
            df[val] = df[val].apply(lambda x: float(x) if str(x).strip().isnumeric() else np.nan)

    return dfs


def transform_data(
    dfs: t.Dict[str, pd.DataFrame],
    quant: pd.DataFrame
) -> t.Dict[str, pd.DataFrame]:
    typer.echo("Transforming data")
    
    dfs_new = {}
    for name, df in tqdm.tqdm(dfs.items()):
        df_tmp = pd.DataFrame()
        
        # scale down and interpolate
        df = df.sort_values(by="date")
        df_tmp[ts_cols] = df[ts_cols]/quant.loc[0.95]
        df_tmp[ts_cols] = df_tmp[ts_cols].interpolate(method="linear")
        df_tmp["date"] = df["date"]
        
        df_features = df_tmp.iloc[-300:]
        dfs_new[name] = df_features
    
    return dfs_new


def get_predictions(
    dfs: t.Dict[str, pd.DataFrame],
    model: models.Model,
    n_pred: int = 50,
) -> t.Dict[str, pd.DataFrame]:
    typer.echo("Generating predictions")

    dfs_new = {}
    for name, df_og in tqdm.tqdm(dfs.items()):
        df = df_og.copy()
        arr = df[-n_pred:][ts_cols].values
        
        for i in range(100):
            arr = np.append(
                arr,
                model.predict(arr[-timesteps:].reshape(1, timesteps, INPUT_LEN), verbose=False),
                axis=0
            )

        df_tmp = pd.DataFrame(columns=ts_cols, data=arr)
        df_tmp["date"] = df["date"]
        dfs_new[name] = df_tmp

    return dfs_new


def save_data(
    dfs: t.Dict[str, pd.DataFrame],
    dfs_pred: t.Dict[str, pd.DataFrame],
) -> None:
    typer.echo("Saving predictions")

    for name, df in tqdm.tqdm(dfs.items()):
        df.to_csv(f"{final_data_dir}/{name}_true.csv")

    for name, df in tqdm.tqdm(dfs_pred.items()):
        df.to_csv(f"{final_data_dir}/{name}_pred.csv")


def main():
    dfs = load_data()
    quant = pd.read_csv(f"{models_dir}/quant.csv", index_col=0)
    dfs = transform_data(dfs, quant)
    model = models.load_model(f"{models_dir}/model.h5")
    dfs_pred = get_predictions(dfs, model)
    save_data(dfs, dfs_pred)


if __name__ == "__main__":
    typer.run(main)
