import json
from pathlib import Path
import random
import sys
import numpy as np
import ch03_select
from ch04_compare import base_table
from ch04_regression import select_renovation
from training_csv import load_training_csv

FOLDS = 5
SEED = 20261011  # データの分割と同じ乱数の種


def table(data, renovation):
    # ch04_regression.py と同じく、説明変数の表に、必要なら改装の列を足す。
    result = base_table(data)
    if renovation:
        result.insert(4, "renovation", (data["renovation"] == "改装済み").astype(float))
    return result


def errors_on_fold(matrix, data, fold):
    # fold以外の取引で係数を求め、foldの取引で絶対パーセント誤差を計算する。
    train = matrix.drop(fold)
    target = np.log(data["price_yen"].astype(float) / data["area"].astype(float))
    beta, _, rank, _ = np.linalg.lstsq(train.to_numpy(), target.drop(fold).to_numpy(), rcond=None)
    if rank != train.shape[1] or not np.isfinite(beta).all():
        raise ValueError("係数が一つに決まりません")
    total = np.exp(matrix.loc[fold].to_numpy() @ beta) * data.loc[fold, "area"].astype(float).to_numpy()
    price = data.loc[fold, "price_yen"].astype(float).to_numpy()
    return list(np.abs(total - price) / price)


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, _ = ch03_select.select_training(load_training_csv(filename))
    data = select_renovation(data)
    order = list(data.index)
    random.Random(SEED).shuffle(order)
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    result = {}
    for renovation, label in ((False, "改装なし"), (True, "改装あり")):
        matrix = table(data, renovation)
        rates = []
        for fold in folds:
            rates += errors_on_fold(matrix, data, fold)
        result[label] = dict(count=len(rates), median=round(float(np.median(rates)), 4),
                             p90=round(float(np.quantile(rates, 0.9)), 4))
    print(json.dumps(result, ensure_ascii=False, indent=2))
