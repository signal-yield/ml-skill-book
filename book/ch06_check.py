import json
from pathlib import Path
import sys
import numpy as np
from ch03_select import select_training
from ch04_regression import fit_model, select_renovation
from training_csv import load_training_csv


def numeric_table(data):
    # 数値で採用した三つの説明変数を、数の列として取り出す。
    age = data["year"].astype(int) - data["built_year"].str.removesuffix("年").astype(int)
    return {"distance": data["distance"].astype(float), "age": age.astype(float), "area": data["area"].astype(float)}


def vif(data, model):
    # 説明変数を一つずつ、残りの説明変数で回帰し、その決定係数から分散拡大係数を求める。
    columns = model["columns"]
    values = numeric_table(data)
    matrix = np.column_stack([np.ones(len(data))] + [values[c] if c in values else
                              (data["renovation"] == "改装済み").astype(float) if c == "renovation" else
                              (data["station" if c.startswith("station") else "year"] == c.split(":")[1]).astype(float)
                              for c in columns[1:]])
    result = {}
    for j, name in enumerate(columns[1:4], start=1):
        others = np.delete(matrix, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, matrix[:, j], rcond=None)
        r2 = 1 - np.sum((matrix[:, j] - others @ beta) ** 2) / np.sum((matrix[:, j] - matrix[:, j].mean()) ** 2)
        result[name] = 1 / (1 - r2)
    return result


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, _ = select_training(load_training_csv(filename))
    data = select_renovation(data)
    model = fit_model(data)
    values = numeric_table(data)
    n, k = len(data), len(model["columns"]) - 1
    r2 = model["training_r2_log_unit_price"]
    candidates = {name: dict(kinds=int(data[name].nunique()), blank=int((data[name] == "").sum()),
                             top=data[name].value_counts().head(3).to_dict())
                  for name in ("layout", "structure", "district", "use", "zoning", "coverage", "floor_area_ratio")}
    print(json.dumps(dict(
        ranges={name: dict(min=float(v.min()), max=float(v.max())) for name, v in values.items()},
        correlation={f"{a}-{b}": float(np.corrcoef(values[a], values[b])[0, 1])
                     for a, b in (("distance", "age"), ("distance", "area"), ("age", "area"))},
        vif=vif(data, model),
        r2=r2, adjusted_r2=1 - (1 - r2) * (n - 1) / (n - k - 1), n=n, explanatory=k,
        candidates=candidates), ensure_ascii=False, indent=2))
