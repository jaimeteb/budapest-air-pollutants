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
data_days = 300
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
    dfs_new = {}

    # formatting of data
    for name, df_og in tqdm.tqdm(dfs.items()):
        df = df_og.copy()

        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date", ascending=True)
        df = df.set_index("date")

        for val in ts_cols:
            df[val] = df[val].apply(
                lambda x: float(x) if str(x).strip().isnumeric() else np.nan
            )

        dfs_new[name] = df

    return dfs_new


def transform_data(
    dfs: t.Dict[str, pd.DataFrame], quant: pd.DataFrame
) -> t.Dict[str, pd.DataFrame]:
    typer.echo("Transforming data")

    dfs_new = {}
    for name, df in tqdm.tqdm(dfs.items()):
        df_tmp = pd.DataFrame()

        # scale down and interpolate
        df_tmp[ts_cols] = df[ts_cols] / quant.loc[0.95]
        df_tmp[ts_cols] = df_tmp[ts_cols].interpolate(method="linear")

        df_features = df_tmp.iloc[-data_days:]
        dfs_new[name] = df_features

    return dfs_new


def get_predictions(
    dfs: t.Dict[str, pd.DataFrame],
    model: models.Model,
    quant: pd.DataFrame,
    n_pred: int = 50,
) -> t.Dict[str, pd.DataFrame]:
    typer.echo("Generating predictions")

    dfs_new = {}
    dfs_joined = {}
    for name, df_og in tqdm.tqdm(dfs.items()):
        df = df_og.copy()[:-n_pred]
        arr = df[ts_cols].values

        for i in range(n_pred):
            arr = np.append(
                arr,
                model.predict(
                    arr[-timesteps:].reshape(1, timesteps, INPUT_LEN), verbose=False
                ),
                axis=0,
            )

        df_tmp = pd.DataFrame(columns=ts_cols, data=arr, index=df_og.index.copy())
        dfs_new[name] = df_tmp

        df_tmp[ts_cols] = df_tmp[ts_cols] * quant.loc[0.95]
        df_og_tmp = df_og.copy()
        df_og_tmp[ts_cols] = df_og_tmp[ts_cols] * quant.loc[0.95]

        dfs_joined[name] = df_og_tmp.join(
            df_tmp[-n_pred:], on="date", rsuffix="_predicted"
        )

    return dfs_new, dfs_joined


def save_data(
    dfs: t.Dict[str, pd.DataFrame],
    dfs_pred: t.Dict[str, pd.DataFrame],
    dfs_joined: t.Dict[str, pd.DataFrame],
) -> None:
    typer.echo("Saving predictions")

    # for name, df in tqdm.tqdm(dfs.items()):
    #     df.to_csv(f"{final_data_dir}/{name}_true.csv", index=True)

    # for name, df in tqdm.tqdm(dfs_pred.items()):
    #     df.to_csv(f"{final_data_dir}/{name}_pred.csv", index=True)

    for name, df in tqdm.tqdm(dfs_joined.items()):
        df.to_csv(f"{final_data_dir}/{name}_joined.csv", index=True)


def main():
    dfs = load_data()
    quant = pd.read_csv(f"{models_dir}/quant.csv", index_col=0)
    dfs = transform_data(dfs, quant)
    model = models.load_model(f"{models_dir}/model.h5")
    dfs_pred, dfs_joined = get_predictions(dfs, model, quant)
    save_data(dfs, dfs_pred, dfs_joined)


if __name__ == "__main__":
    typer.run(main)
