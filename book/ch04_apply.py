import json
from pathlib import Path
import sys
import numpy as np
from ch03_select import select_training
from ch04_regression import fit_model, select_renovation
from training_csv import load_training_csv


def apply_model(model, room):
    if room.get("renovation") not in ("未改装", "改装済み"):
        raise ValueError(f"改装「{room.get('renovation', '')}」は、「未改装」か「改装済み」で入力してください")
    # 回帰と同じ列の並びで、部屋の値を0と1や数値に置き換える。
    values = dict(intercept=1.0, distance=room["distance"], age=room["year"] - room["built_year"], area=room["area"],
                  renovation=1.0 if room["renovation"] == "改装済み" else 0.0)
    values["station:" + room["station"]] = 1.0
    values["year:" + str(room["year"])] = 1.0
    known = set(model["columns"]) | {"station:" + model["reference"]["station"], "year:2025"}
    if not set(values) <= known:
        raise ValueError("モデルにない駅や年です")
    row = np.array([values.get(column, 0.0) for column in model["columns"]])
    unit = float(np.exp(row @ np.array(model["coefficients"])))
    return dict(room, predicted_yen_per_m2=unit, predicted_total_yen=unit * room["area"])


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data, _ = select_training(load_training_csv(filename))
    data = select_renovation(data)
    model = fit_model(data)
    rooms = [dict(room="A", station="用賀", distance=6, area=70, built_year=2005, renovation="未改装", year=2025),
             dict(room="B", station="用賀", distance=7, area=72, built_year=2004, renovation="未改装", year=2025)]
    print(json.dumps([apply_model(model, room) for room in rooms], ensure_ascii=False, indent=2))
