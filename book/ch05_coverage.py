"""スキルがテストデータの物件のうち、何件に答えられるか（保留率）を数える。価格の列は読み込まない。"""
from collections import Counter
import json
from pathlib import Path
import sys

import ch03_select
from ch04_regression import select_renovation
from training_csv import load_test_inputs

SKILL = Path(__file__).resolve().parent.parent / "skill" / "mansion-price" / "scripts"
sys.path.insert(0, str(SKILL))
import predict  # noqa: E402


def to_room(row):
    # スキルに渡す形にする。取引年は、スキルの使い方どおり2025年とする。
    return dict(station=row["station"], distance=row["distance"], area=row["area"],
                built_year=row["built_year"].removesuffix("年"), renovation=row["renovation"], year=2025)


def label(reasons):
    # 一件ごとに、保留の理由を一つの分類にまとめる。
    text = reasons[0]
    if text.startswith("入力が足りません："):
        return "入力が足りない（" + text.split("：")[1] + "）"
    if text.startswith("数値にできません"):
        return "数値にできない入力"
    if "の外です" in text:
        return "範囲の外（" + text.split("が")[0] + "）"
    return "その他"


def coverage(data, model):
    reasons = Counter()
    answered = []
    for key, row in data.iterrows():
        found = predict.check(to_room(row), model)
        if found:
            reasons[label(found)] += 1
        else:
            answered.append(key)
    total = len(data)
    return dict(test_rows=total, answered=len(answered), withheld=total - len(answered),
                withheld_rate=round((total - len(answered)) / total, 4),
                withheld_reasons=dict(reasons.most_common())), answered


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    data = load_test_inputs(filename)
    model = json.loads(predict.MODEL.read_text(encoding="utf-8"))
    result, answered = coverage(data, model)
    # 第5章で誤差を数えた取引（規則1〜4と6を通った取引）と、答えられた取引が同じかを、価格を使わずに確かめる。
    evaluated = select_renovation(ch03_select.select_basic(data)[0]).index
    result["same_as_evaluated_rows"] = set(answered) == set(evaluated)
    print(json.dumps(dict(result, audit=data.attrs["audit"]), ensure_ascii=False, indent=2))
