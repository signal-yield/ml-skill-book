"""選別した取引で散布図を描く。規則1〜4の段階と、規則5まで適用した段階の2枚。"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ch03_select
from training_csv import load_training_csv

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "sans-serif"]


def draw(data, title, path):
    unit = data["price_yen"].astype(float) / data["area"].astype(float) / 10000
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["distance"].astype(float), unit, s=10, alpha=0.6)
    ax.set_xlabel("駅からの徒歩分数（分）")
    ax.set_ylabel("㎡単価（万円）")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    top = unit.idxmax()
    return dict(plotted_count=len(data), max_unit_price_man_yen=round(float(unit[top]), 1),
                area_of_max_m2=float(data.loc[top, "area"]))


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    source = load_training_csv(filename)
    multiplier = ch03_select.IQR_MULTIPLIER
    ch03_select.IQR_MULTIPLIER = float("inf")  # 規則5を実質的に外し、規則1〜4の段階を描く
    before, flow_before = ch03_select.select_training(source)
    ch03_select.IQR_MULTIPLIER = multiplier
    after, flow_after = ch03_select.select_training(source)
    result = dict(rules_1_to_4=dict(flow=flow_before[:-1], **draw(before, "規則1〜4を適用したトレーニングデータ（619件）", out / "scatter_rules1to4.png")),
                  rules_1_to_5=dict(flow=flow_after, **draw(after, "規則1〜5を適用したトレーニングデータ（618件）", out / "scatter_rules1to5.png")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
