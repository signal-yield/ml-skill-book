import json
from pathlib import Path
import sys
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "assets" / "model.json"
NOT_IN_DATA = ("所在階", "方位", "管理の状態", "室内の状態")
INPUT_NAMES = dict(station="最寄駅", distance="徒歩分数", area="面積", built_year="建築年", renovation="改装", year="取引年")


def numeric_input(room):
    result, reasons = dict(room), []
    for name in ("distance", "area", "built_year", "year"):
        value = room[name]
        try:
            number = float(value)
            if isinstance(value, bool) or not np.isfinite(number):
                raise ValueError
            if name in ("built_year", "year"):
                if not number.is_integer():
                    raise ValueError
                number = int(number)
            result[name] = number
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"数値にできません：{INPUT_NAMES[name]}「{value}」")
    return result, reasons


def check(room, model):
    # 予測の前に、入力が欠けていないか、データの範囲に入っているかを確かめる。
    missing = [name for name in ("station", "distance", "area", "built_year", "renovation", "year")
               if room.get(name) in (None, "")]
    if missing:
        return ["入力が足りません：" + "、".join(INPUT_NAMES[name] for name in missing)]
    room, reasons = numeric_input(room)
    if room["renovation"] not in ("未改装", "改装済み"):
        reasons.append(f"改装「{room['renovation']}」は、「未改装」か「改装済み」で入力してください")
    if reasons:
        return reasons
    if room["year"] != 2025:
        return [f"取引年{room['year']}年は受け付けていません。2025年の取引価格の水準で予測するため、2025年を指定してください"]
    room = dict(room, age=room["year"] - room["built_year"])
    for name, label in (("distance", "徒歩分数"), ("age", "築年数"), ("area", "面積")):
        low, high = model["ranges"][name]["min"], model["ranges"][name]["max"]
        if not low <= float(room[name]) <= high:
            reasons.append(f"{label}が{room[name]}で、データの範囲（{low:g}〜{high:g}）の外です")
    if room["station"] not in model["stations"]:
        reasons.append(f"最寄駅「{room['station']}」は、データに含まれていません")
    if str(room["year"]) not in model["years"]:
        reasons.append(f"取引年{room['year']}年は、データに含まれていません")
    return reasons


def predict(room, model):
    reasons = check(room, model)
    if reasons:
        raise ValueError("、".join(reasons))
    room, _ = numeric_input(room)
    # 回帰と同じ並びで値を置き、係数を掛けて合計し、㎡単価の予測値に戻す。
    values = dict(intercept=1.0, distance=float(room["distance"]),
                  age=float(int(room["year"]) - int(room["built_year"])), area=float(room["area"]),
                  renovation=1.0 if room["renovation"] == "改装済み" else 0.0)
    values["station:" + room["station"]] = 1.0
    values["year:" + str(room["year"])] = 1.0
    row = np.array([values.get(column, 0.0) for column in model["columns"]])
    unit = float(np.exp(row @ np.array(model["coefficients"])))
    return unit, unit * float(room["area"])


def man_yen(value):
    # 円を「約◯億◯万円」の形にする（1万円未満は四捨五入）。
    man = round(value / 10000)
    return f"約{man // 10000}億{man % 10000}万円" if man >= 10000 else f"約{man}万円"


def answer(room, model):
    reasons = check(room, model)
    ranges = model["ranges"]
    scope = (f"予測できる範囲は、最寄駅が{'、'.join(model['stations'])}、"
             f"徒歩{ranges['distance']['min']:g}〜{ranges['distance']['max']:g}分、"
             f"築{ranges['age']['min']:g}〜{ranges['age']['max']:g}年、"
             f"面積{ranges['area']['min']:g}〜{ranges['area']['max']:g}㎡の物件です。")
    if reasons:
        return dict(status="予測できません", reasons=reasons, scope=scope)
    unit, total = predict(room, model)
    error = model["test_error"]
    return dict(status="予測しました", input=room,
                predicted_yen_per_m2=round(unit), predicted_total_yen=round(total),
                predicted_total=man_yen(total),
                meaning=("2025年の取引価格の水準で、この要因の物件が取引されたときの価格の予測値です。"
                         "売り出しの価格や、いま売れる価格を示すものではありません。"),
                error=(f"モデルづくりに使っていない{error['count']}件の取引では、予測値と実際の取引価格の差が、"
                       f"半分は{error['median']:.2%}以内、10件のうち9件は{error['p90']:.2%}以内でした。"
                       "この誤差は、差の大きさを実際の取引価格に対する割合で表したものです。"
                       "高く予測したか低く予測したかは区別していません。"),
                not_in_data=("次の価格形成要因はデータに含まれていないため、予測値に反映されていません："
                             + "、".join(NOT_IN_DATA)),
                scope=scope)


if __name__ == "__main__":
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    room = json.loads(sys.argv[1])
    print(json.dumps(answer(room, model), ensure_ascii=False, indent=2))
