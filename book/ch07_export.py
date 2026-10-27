import json
from pathlib import Path
import sys
from ch03_select import select_training
from ch04_regression import fit_model, select_renovation
from ch06_check import numeric_table
from training_csv import load_training_csv, STATIONS

if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("model.json")
    data, _ = select_training(load_training_csv(filename))
    data = select_renovation(data)
    model = fit_model(data)
    values = numeric_table(data)
    # 予測に必要な値だけを書き出す。取引一件ごとの価格は書き出さない。
    saved = dict(columns=model["columns"], coefficients=model["coefficients"], reference=model["reference"],
                 stations=list(STATIONS), years=["2021", "2022", "2023", "2024", "2025"],
                 ranges={name: dict(min=float(v.min()), max=float(v.max())) for name, v in values.items()},
                 # テストデータの誤差は、テストデータで評価したときの記録の値を写す。テストデータはここでは読まない。
                 test_error=dict(count=143, median=0.12831209203903424, p90=0.3381816175219381),
                 n=model["n"])
    output.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(written=str(output.name), n=saved["n"], columns=len(saved["columns"])), ensure_ascii=False, indent=2))
