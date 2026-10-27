"""第2章の散布図：トレーニングデータの6駅1,250件を、選別の前に描く。"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from training_csv import load_training_csv

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "sans-serif"]

if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    data = load_training_csv(filename)
    minutes = pd.to_numeric(data["distance"].where(data["distance"].str.fullmatch(r"[0-9]+")), errors="coerce")
    area = pd.to_numeric(data["area"].where(data["area"].str.fullmatch(r"[0-9]+(?:\.[0-9]+)?")), errors="coerce")
    unit = data["price_yen"].astype(float) / area
    keep = minutes.notna() & area.notna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(minutes[keep], unit[keep] / 10000, s=10, alpha=0.6)
    ax.set_xlabel("最寄駅からの徒歩分数（分）")
    ax.set_ylabel("㎡単価（万円）")
    ax.set_title(f"トレーニングデータ（6駅、{len(data)}件のうち{int(keep.sum())}点）")
    fig.tight_layout()
    fig.savefig(out / "scatter_ch02.png", dpi=150)
    six = unit[keep & (minutes == 6)]
    print(json.dumps(dict(plotted_count=int(keep.sum()), excluded_numeric_count=int((~keep).sum()),
                          excluded_numeric_reasons={"徒歩が数字でない": int(minutes.isna().sum()),
                                                    "面積が数字でない": int(area.isna().sum())},
                          six_minutes=dict(count=len(six), min_yen_per_m2=float(six.min()),
                                           median_yen_per_m2=float(six.median()), max_yen_per_m2=float(six.max()),
                                           max_div_min=float(six.max() / six.min()))),
                     ensure_ascii=False, indent=2))
