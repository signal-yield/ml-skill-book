"""固定した教材CSVから、トレーニングデータの6駅だけを安全に読み込む。"""
from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import random

import pandas as pd

SOURCE_SHA256 = "36f5d00a50f34d8c1bcbfc7c35e617604ec119e4fb986b991b32c88a2aabaadd"
SOURCE_COUNT = 11266
STATIONS = ("池尻大橋", "三軒茶屋", "駒沢大学", "桜新町", "用賀", "二子玉川")
ALLOWED = {"トレーニングデータ"}
OLD_TO_NEW = {"学習用": "トレーニングデータ", "調整用": "トレーニングデータ", "最終確認用": "テストデータ"}
FIELDS = {"価格情報区分": "information_type", "種類": "kind", "都道府県名": "prefecture",
          "市区町村名": "city", "最寄駅：名称": "station", "最寄駅：距離（分）": "distance",
          "面積（㎡）": "area", "建築年": "built_year", "取引時期": "year",
          "間取り": "layout", "建物の構造": "structure", "改装": "renovation", "地区名": "district",
          "用途": "use", "都市計画": "zoning", "建ぺい率（％）": "coverage", "容積率（％）": "floor_area_ratio"}


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fixed_labels():
    order = list(range(1, SOURCE_COUNT + 1))
    random.Random(20261011).shuffle(order)
    return {key: "トレーニングデータ" if i < SOURCE_COUNT * 80 // 100 else "テストデータ" for i, key in enumerate(order)}


def allowed_lines(source, selected, labels, allowed=ALLOWED):
    if any(labels.get(key) not in allowed for key in selected):
        raise PermissionError("テストデータまたは不明なラベルの価格は読めません")
    pieces, ids, count = [], [], 0
    with Path(source).open("rb") as handle:
        pieces.append(handle.readline())
        for key, line in enumerate(handle, 1):
            count = key
            if key in selected:
                pieces.append(line)
                ids.append(key)
    if count != SOURCE_COUNT or set(ids) != set(selected):
        raise ValueError("元の行番号と教材CSVの行数が一致しません")
    return b"".join(pieces), ids


def read_columns(raw, columns):
    for encoding in ("cp932", "utf-8-sig"):
        try:
            text = raw.decode(encoding)
            return pd.read_csv(io.StringIO(text), usecols=columns, dtype=str, keep_default_na=False), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("教材CSVの文字コードを確認してください")


def load_training_csv(source, labels_path=None):
    source = Path(source)
    before = fingerprint(source)
    if before != SOURCE_SHA256:
        raise ValueError("教材CSVの指紋が一致しません。別の取得版へ自動で切り替えません")
    labels = fixed_labels()
    if labels_path is not None:
        with Path(labels_path).open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        saved = {int(row["source_row"]): OLD_TO_NEW[row["label"]] for row in records}
        if len(records) != len(saved) or saved != labels:
            raise ValueError("既存の固定ラベルと再現したラベルが一致しません")
    # ラベルを確定し、テストデータの行を読み込みの対象から除く。
    raw, ids = allowed_lines(source, {key for key in labels if labels[key] in ALLOWED}, labels)
    data, encoding = read_columns(raw, list(FIELDS))
    data = data.rename(columns=FIELDS)
    data.index = pd.Index(ids, name="source_row")
    data["year"] = data["year"].str.extract(r"^(202[1-5])年第[1-4]四半期$", expand=False).fillna("")
    data = data.loc[(data["information_type"] == "不動産取引価格情報") & (data["kind"] == "中古マンション等")
                    & (data["prefecture"] == "東京都") & (data["city"] == "世田谷区")
                    & data["station"].isin(STATIONS) & (data["year"] != "")].copy()
    if len(data) != 1250:
        raise ValueError("6駅のトレーニングデータの件数が1250件と一致しません")
    raw, ids = allowed_lines(source, set(data.index), labels)
    prices, price_encoding = read_columns(raw, ["取引価格（総額）"])
    if encoding != price_encoding or len(prices) != len(data) or ids != list(data.index):
        raise ValueError("価格と非価格列の行対応が異なります")
    data["price_yen"] = prices["取引価格（総額）"].to_numpy()
    data["label"] = [labels[key] for key in data.index]
    if fingerprint(source) != before:
        raise ValueError("実行中に元CSVの指紋が変わりました")
    data.attrs["audit"] = dict(source_filename=source.name, source_sha256=before, encoding=encoding,
                               seed=20261011, fixed_labels_match=labels_path is not None,
                               price_rows_loaded=len(prices), test_price_values_read=0,
                               training_labels=dict(Counter(data["label"])),
                               method="固定ラベルでテストデータを除外してから読み込む。価格列を取り込むのはトレーニングデータの6駅のみ")
    return data


def load_test_csv(source, labels_path=None):
    """テストデータの6駅を読み込む。第5章で一度だけ使う。"""
    source = Path(source)
    before = fingerprint(source)
    if before != SOURCE_SHA256:
        raise ValueError("教材CSVの指紋が一致しません。別の取得版へ自動で切り替えません")
    labels = fixed_labels()
    if labels_path is not None:
        with Path(labels_path).open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        saved = {int(row["source_row"]): OLD_TO_NEW[row["label"]] for row in records}
        if len(records) != len(saved) or saved != labels:
            raise ValueError("既存の固定ラベルと再現したラベルが一致しません")
    # ラベルを確定し、テストデータの行だけを読み込む。
    raw, ids = allowed_lines(source, {key for key in labels if labels[key] == "テストデータ"}, labels, {"テストデータ"})
    data, encoding = read_columns(raw, list(FIELDS))
    data = data.rename(columns=FIELDS)
    data.index = pd.Index(ids, name="source_row")
    data["year"] = data["year"].str.extract(r"^(202[1-5])年第[1-4]四半期$", expand=False).fillna("")
    data = data.loc[(data["information_type"] == "不動産取引価格情報") & (data["kind"] == "中古マンション等")
                    & (data["prefecture"] == "東京都") & (data["city"] == "世田谷区")
                    & data["station"].isin(STATIONS) & (data["year"] != "")].copy()
    if len(data) != 316:
        raise ValueError("6駅のテストデータの件数が316件と一致しません")
    raw, ids = allowed_lines(source, set(data.index), labels, {"テストデータ"})
    prices, price_encoding = read_columns(raw, ["取引価格（総額）"])
    if encoding != price_encoding or len(prices) != len(data) or ids != list(data.index):
        raise ValueError("価格と非価格列の行対応が異なります")
    data["price_yen"] = prices["取引価格（総額）"].to_numpy()
    data["label"] = [labels[key] for key in data.index]
    if fingerprint(source) != before:
        raise ValueError("実行中に元CSVの指紋が変わりました")
    data.attrs["audit"] = dict(source_filename=source.name, source_sha256=before, encoding=encoding,
                               seed=20261011, fixed_labels_match=labels_path is not None,
                               price_rows_loaded=len(prices), test_price_values_read=len(prices),
                               test_labels=dict(Counter(data["label"])),
                               method="固定ラベルでテストデータの行だけを読み込む。第5章の評価で一度だけ使う")
    return data
