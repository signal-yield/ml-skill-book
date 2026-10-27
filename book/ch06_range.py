def out_of_range(room, ranges, stations, years):
    # 予測の前に、入力がトレーニングデータの範囲に入っているかを確かめ、範囲の外にある理由を並べて返す。
    reasons = []
    if "built_year" in room:
        room = dict(room, age=int(room["year"]) - int(room["built_year"]))
    for name, label in (("distance", "徒歩分数"), ("age", "築年数"), ("area", "面積")):
        low, high = ranges[name]["min"], ranges[name]["max"]
        if not low <= room[name] <= high:
            reasons.append(f"{label}が{room[name]}で、データの範囲（{low:g}〜{high:g}）の外です")
    if room["station"] not in stations:
        reasons.append(f"最寄駅「{room['station']}」は、データに含まれていません")
    if str(room["year"]) not in years:
        reasons.append(f"取引年{room['year']}年は、データに含まれていません")
    return reasons
