import json
from pathlib import Path
import sys
import pandas as pd
from training_csv import load_training_csv

IQR_MULTIPLIER = 3  # 異常値の判定に使う、四分位範囲の倍率


def select_basic(data):
    # 規則1〜4：6駅の取引から、面積、建築年、徒歩分数の順に選ぶ。
    data = data.copy()
    flow = [("6駅（トレーニングデータ）", len(data))]
    area = pd.to_numeric(data["area"].where(data["area"].str.fullmatch(r"[0-9]+(?:\.[0-9]+)?")), errors="coerce")
    data = data.loc[area >= 50].copy()
    flow.append(("面積50㎡以上", len(data)))
    built = pd.to_numeric(data["built_year"].str.extract(r"^([0-9]{4})(?:年)?$", expand=False), errors="coerce")
    data = data.loc[built.notna() & (built <= data["year"].astype(int))].copy()
    flow.append(("建築年が分かる", len(data)))
    minutes = pd.to_numeric(data["distance"].where(data["distance"].str.fullmatch(r"[0-9]+")), errors="coerce")
    data = data.loc[minutes.notna()].copy()
    flow.append(("徒歩分数が数値", len(data)))
    return data, flow


def select_training(data):
    # 規則1〜4の後で、規則5の異常値を除外する。
    data, flow = select_basic(data)
    unit = data["price_yen"].astype(float) / data["area"].astype(float)
    q1, q3 = unit.quantile([0.25, 0.75])
    if not q1 < q3:
        raise ValueError("四分位範囲が0なので、異常値を判定できません")
    lower, upper = q1 - IQR_MULTIPLIER * (q3 - q1), q3 + IQR_MULTIPLIER * (q3 - q1)
    data = data.loc[(unit > lower) & (unit < upper)].copy()
    flow.append(("異常値を除外", len(data)))
    data.attrs["outlier_bounds"] = dict(q1=q1, q3=q3, lower=lower, upper=upper)
    return data, flow


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, flow = select_training(load_training_csv(filename))
    print(json.dumps(dict(count=len(data), flow=flow, outlier_bounds=data.attrs["outlier_bounds"]),
                     ensure_ascii=False, indent=2))
