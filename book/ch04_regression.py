import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ch03_select import select_training
from training_csv import load_training_csv, STATIONS


def select_renovation(data):
    # 規則6：改装の項目が空欄の取引を、欠損値として除外する。
    return data.loc[data["renovation"].isin(["未改装", "改装済み"])].copy()


def fit_model(data):
    # 量的な説明変数は数値のまま、改装と駅と取引年は0と1の列にする。
    table = pd.DataFrame(index=data.index)
    table["intercept"] = 1.0
    table["distance"] = data["distance"].astype(float)
    table["age"] = data["year"].astype(int) - data["built_year"].str.removesuffix("年").astype(int)
    table["area"] = data["area"].astype(float)
    table["renovation"] = (data["renovation"] == "改装済み").astype(float)
    station = max(STATIONS, key=lambda name: (data["station"] == name).sum())
    for name in STATIONS:
        if name != station:
            table["station:" + name] = (data["station"] == name).astype(float)
    for year in ("2021", "2022", "2023", "2024"):
        table["year:" + year] = (data["year"] == year).astype(float)
    # 対数の㎡単価を目的変数にして、最小二乗法で係数を求める。
    target = np.log(data["price_yen"].astype(float) / data["area"].astype(float)).to_numpy()
    matrix = table.to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != matrix.shape[1] or not np.isfinite(beta).all():
        raise ValueError("係数が一つに決まりません")
    r2 = 1 - np.sum((matrix @ beta - target) ** 2) / np.sum((target - target.mean()) ** 2)
    counts = {name: int((data["station"] == name).sum()) for name in STATIONS}
    return dict(n=len(data), station_counts=counts, reference=dict(station=station, year="2025"), columns=list(table.columns),
                coefficients=beta.tolist(), ratio=np.exp(beta[1:]).tolist(),
                training_r2_log_unit_price=float(r2))


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, _ = select_training(load_training_csv(filename))
    data = select_renovation(data)
    print(json.dumps(fit_model(data), ensure_ascii=False, indent=2))
