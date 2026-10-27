import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import ch03_select
from training_csv import load_training_csv, STATIONS

CANDIDATES = {"renovation": "改装", "district": "地区名", "zoning": "都市計画", "coverage": "建ぺい率",
              "floor_area_ratio": "容積率", "layout": "間取り", "structure": "建物の構造", "use": "用途"}
NUMERIC = {"coverage", "floor_area_ratio"}


def base_table(data):
    # ch04_regression.py と同じく、徒歩分数、築年数、面積、最寄駅、取引年の表を作る。
    table = pd.DataFrame(index=data.index)
    table["intercept"] = 1.0
    table["distance"] = data["distance"].astype(float)
    table["age"] = data["year"].astype(int) - data["built_year"].str.removesuffix("年").astype(int)
    table["area"] = data["area"].astype(float)
    station = max(STATIONS, key=lambda name: (data["station"] == name).sum())
    for name in STATIONS:
        if name != station:
            table["station:" + name] = (data["station"] == name).astype(float)
    for year in ("2021", "2022", "2023", "2024"):
        table["year:" + year] = (data["year"] == year).astype(float)
    return table


def adjusted_r2(table, target):
    # 最小二乗法で係数を求め、自由度調整済み決定係数を返す。
    matrix = table.to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != matrix.shape[1] or not np.isfinite(beta).all():
        raise ValueError("係数が一つに決まりません")
    r2 = 1 - np.sum((matrix @ beta - target) ** 2) / np.sum((target - target.mean()) ** 2)
    n, k = matrix.shape
    return 1 - (1 - r2) * (n - 1) / (n - k)


def compare(data, name):
    # 項目が空欄でない取引だけで、項目を足す前と足した後の当てはまりを比べる。
    rows = data.loc[data[name] != ""]
    target = np.log(rows["price_yen"].astype(float) / rows["area"].astype(float)).to_numpy()
    table = base_table(rows)
    added = pd.DataFrame(index=rows.index)
    if name in NUMERIC:
        added[name] = rows[name].astype(float)
    else:
        kinds = rows[name].value_counts()
        for kind in kinds.index[1:]:
            added[name + ":" + kind] = (rows[name] == kind).astype(float)
    before, after = adjusted_r2(table, target), adjusted_r2(pd.concat([table, added], axis=1), target)
    return dict(item=CANDIDATES[name], n=len(rows), added=added.shape[1],
                before=round(before, 4), after=round(after, 4), gain=round(after - before, 4))


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, _ = ch03_select.select_training(load_training_csv(filename))
    print(json.dumps([compare(data, name) for name in CANDIDATES], ensure_ascii=False, indent=2))
