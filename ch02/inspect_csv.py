"""CSVの列と件数だけを確認する。価格列の値は表に読み込まない。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

STATIONS = ("用賀", "池尻大橋", "三軒茶屋", "駒沢大学", "桜新町", "二子玉川")
SAFE_COLUMNS = ("種類", "価格情報区分", "都道府県名", "市区町村名", "取引時期",
                "最寄駅：名称", "最寄駅：距離（分）", "建築年", "面積（㎡）", "取引の事情等")


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    with path.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inspect(source, expected_hash):
    digest = fingerprint(source)
    if digest != expected_hash:
        raise ValueError("教材CSVの指紋が一致しません")
    for encoding in ("cp932", "utf-8-sig"):
        try:
            with source.open(encoding=encoding, newline="") as handle:
                columns = next(csv.reader(handle))
            # usecolsにより、価格列の値はデータフレームに取り込まない。
            frame = pd.read_csv(source, encoding=encoding, usecols=list(SAFE_COLUMNS),
                                dtype=str, keep_default_na=False)
            break
        except UnicodeDecodeError:
            if encoding == "utf-8-sig":
                raise
    assert all("価格" not in name or name == "価格情報区分" for name in frame.columns)
    frame.insert(0, "source_row", range(1, len(frame) + 1))
    mansion = frame[frame["種類"] == "中古マンション等"]
    trade = mansion[mansion["価格情報区分"] == "不動産取引価格情報"]
    years = trade["取引時期"].str.extract(r"^(\d{4})年第[1-4]四半期$")[0]
    if years.isna().any() or not years.astype(int).between(2021, 2025).all():
        raise ValueError("取引時期が指定範囲外です")
    if set(frame["都道府県名"]) != {"東京都"} or set(frame["市区町村名"]) != {"世田谷区"}:
        raise ValueError("対象地域が異なります")
    counts = lambda rows: {station: int((rows["最寄駅：名称"] == station).sum()) for station in STATIONS}
    summary = dict(filename=source.name, acquisition_date="2026-10-11（ユーザー確認済み）",
                   region="東京都世田谷区", period="2021年第1四半期〜2025年第4四半期",
                   encoding=encoding, sha256=digest, total_rows=len(frame), columns=columns,
                   mansion_rows=len(mansion), information_type_counts=mansion["価格情報区分"].value_counts().to_dict(),
                   trade_rows=len(trade), station_counts_trade_only=counts(trade),
                   station_counts_all_information_types=counts(mansion), price_columns_loaded=False)
    return summary, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-nonprice", type=Path, required=True)
    args = parser.parse_args()
    book = Path(__file__).resolve().parents[2]
    if args.private_nonprice.resolve().is_relative_to(book.resolve()):
        raise ValueError("行単位の中間ファイルはリポジトリ外に保存してください")
    summary, frame = inspect(args.source, args.expected_sha256)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.private_nonprice, dict(source_sha256=summary["sha256"], rows=frame.to_dict("records")))
    summary["private_nonprice_sha256"] = fingerprint(args.private_nonprice)
    save_json(args.output / "csv_summary.json", summary)
    print(f"全体の件数：{summary['total_rows']:,}（ヘッダーを除く）")
    print("列名：" + "、".join(summary["columns"]))
    print(f"種類＝中古マンション等：{summary['mansion_rows']:,}")
    print("価格情報区分別：" + json.dumps(summary["information_type_counts"], ensure_ascii=False))
    print("以降の分析は不動産取引価格情報だけを使用します。")
    print("6駅の件数（不動産取引価格情報・中古マンション等、面積等の絞り込み前）：")
    for station, count in summary["station_counts_trade_only"].items():
        print(f"{station}：{count:,}")
    print("取得日：" + summary["acquisition_date"])
    print("ファイル名：" + summary["filename"])
    print("文字コード：" + summary["encoding"])
    print("SHA256：" + summary["sha256"])


if __name__ == "__main__":
    main()
