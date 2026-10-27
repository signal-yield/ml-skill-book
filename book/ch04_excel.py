"""第4章の重回帰を、Excel の LINEST 関数でも計算できるように、トレーニングデータの566件を xlsx に書き出す。"""
import json
from pathlib import Path
import sys

from openpyxl import Workbook
from openpyxl.worksheet.formula import ArrayFormula

import ch03_select
from ch04_regression import fit_model, select_renovation
from training_csv import load_training_csv, STATIONS

OTHER_STATIONS = ["池尻大橋", "三軒茶屋", "駒沢大学", "桜新町", "二子玉川"]  # 基準の用賀は列を作らない
YEARS = ["2021", "2022", "2023", "2024"]  # 基準の2025年は列を作らない
NAMES = ["徒歩分", "築年数", "面積", "改装"] + OTHER_STATIONS + YEARS


def build(data, model, output):
    book = Workbook()
    sheet = book.active
    sheet.title = "データ"
    head = ["最寄駅", "徒歩分", "面積", "建築年", "取引年", "改装", "取引価格", "㎡単価の対数"] + NAMES
    sheet.append(head)
    for i, (_, row) in enumerate(data.iterrows(), start=2):
        built = int(row["built_year"].removesuffix("年"))
        values = [row["station"], int(row["distance"]), float(row["area"]), built, int(row["year"]),
                  row["renovation"], int(row["price_yen"])]
        formulas = [f"=LN(G{i}/C{i})", f"=B{i}", f"=E{i}-D{i}", f"=C{i}", f'=IF(F{i}="改装済み",1,0)']
        formulas += [f'=IF(A{i}="{name}",1,0)' for name in OTHER_STATIONS]
        formulas += [f"=IF(E{i}={year},1,0)" for year in YEARS]
        sheet.append(values + formulas)
    last = len(data) + 1
    result = book.create_sheet("LINEST")
    # LINEST は、係数を「最後の列から順に」並べ、最後に切片を返す。
    result.append(["LINEST の結果（1行目が係数。左から、" + "、".join(reversed(NAMES)) + "、切片の順）"])
    result["A2"] = ArrayFormula("A2:N6", f"=LINEST(データ!H2:H{last},データ!I2:U{last},TRUE,TRUE)")
    result["A8"] = "説明変数"
    result["B8"] = "LINEST の係数"
    result["C8"] = "Python（第4章）の係数"
    result["D8"] = "差"
    python = dict(zip(model["columns"], model["coefficients"]))
    keys = ["distance", "age", "area", "renovation"] + ["station:" + s for s in OTHER_STATIONS] + ["year:" + y for y in YEARS]
    for j, (name, key) in enumerate(zip(NAMES + ["切片"], keys + ["intercept"]), start=9):
        column = 14 if key == "intercept" else 13 - (j - 9)  # LINEST の並びは逆順。切片は14列目
        letter = "ABCDEFGHIJKLMN"[column - 1]
        result[f"A{j}"] = name
        result[f"B{j}"] = f"=LINEST!{letter}2"
        result[f"C{j}"] = python[key]
        result[f"D{j}"] = f"=B{j}-C{j}"
    result["A23"] = "決定係数（R²）"
    result["B23"] = "=LINEST!A4"
    book.save(output)


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Tokyo_Setagaya Ward_20211_20254.csv")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ch04_linest.xlsx")
    data, _ = ch03_select.select_training(load_training_csv(filename))
    data = select_renovation(data)
    model = fit_model(data)
    build(data, model, output)
    print(json.dumps(dict(written=output.name, rows=len(data), columns=len(NAMES)), ensure_ascii=False, indent=2))
