import json
from pathlib import Path
import sys
import numpy as np
import ch03_select
from ch04_regression import fit_model, select_renovation
from training_csv import load_training_csv, load_test_csv


def predict(model, data):
    # 回帰と同じ列を作り、係数を掛けて合計し、㎡単価の予測値に戻す。
    values = {"intercept": np.ones(len(data)), "distance": data["distance"].astype(float).to_numpy(),
              "age": (data["year"].astype(int) - data["built_year"].str.removesuffix("年").astype(int)).to_numpy(),
              "area": data["area"].astype(float).to_numpy(),
              "renovation": (data["renovation"] == "改装済み").astype(float).to_numpy()}
    for column in model["columns"][5:]:
        group, name = column.split(":")
        values[column] = (data["station" if group == "station" else "year"] == name).astype(float).to_numpy()
    matrix = np.column_stack([values[column] for column in model["columns"]])
    return np.exp(matrix @ np.array(model["coefficients"]))


def errors(unit_estimate, data):
    # 絶対パーセント誤差＝|総額の予測値−取引価格|÷取引価格。中央値と90パーセンタイルを出す。
    price = data["price_yen"].astype(float).to_numpy()
    rate = np.abs(unit_estimate * data["area"].astype(float).to_numpy() - price) / price
    return dict(count=len(data), median=float(np.median(rate)), p90=float(np.quantile(rate, 0.9, method="linear")))


def evaluate(filename):
    train, _ = ch03_select.select_training(load_training_csv(filename))
    train = select_renovation(train)
    model = fit_model(train)
    mean_unit = float((train["price_yen"].astype(float) / train["area"].astype(float)).mean())
    # テストデータは一度だけ読み、規則1〜4と6を適用する。価格で選ぶ規則5は適用しない。
    test, flow = ch03_select.select_basic(load_test_csv(filename))
    flow = [list(step) for step in flow]
    test = select_renovation(test)
    flow.append(["改装が分かる", len(test)])
    flow[0][0] = "6駅（テストデータ）"
    return dict(test_flow=flow, test_rule5_applied=False,
                model_on_training=errors(predict(model, train), train),
                model_on_test=errors(predict(model, test), test),
                mean_unit_on_test=errors(np.full(len(test), mean_unit), test),
                training_mean_unit_yen=mean_unit)


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    print(json.dumps(evaluate(filename), ensure_ascii=False, indent=2))
