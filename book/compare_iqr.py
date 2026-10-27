"""異常値の判定の倍率（3倍と1.5倍）で、除外される取引の数を駅ごとに比べる。個々の価格は出さない。"""
import json
from pathlib import Path
import sys
import ch03_select
from training_csv import load_training_csv, STATIONS

if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    source = load_training_csv(filename)
    ch03_select.IQR_MULTIPLIER = float("inf")
    base, _ = ch03_select.select_training(source)
    unit = base["price_yen"].astype(float) / base["area"].astype(float)
    result = {}
    for multiplier in (3, 1.5):
        ch03_select.IQR_MULTIPLIER = multiplier
        kept, _ = ch03_select.select_training(source)
        out = base.loc[~base.index.isin(kept.index)]
        out_unit = unit.loc[out.index] / 10000
        result[str(multiplier)] = dict(base_count=len(base), excluded=len(out), upper_man_yen=round(kept.attrs["outlier_bounds"]["upper"] / 10000, 1),
                                       by_station={name: int((out["station"] == name).sum()) for name in STATIONS},
                                       excluded_min_man_yen=round(float(out_unit.min()), 1),
                                       excluded_second_max_man_yen=round(float(out_unit.sort_values().iloc[-2]), 1) if len(out) > 1 else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
