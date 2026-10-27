import ast
from pathlib import Path
import sys


def functions(filename):
    # ファイルを読み、関数の名前と、その関数のコードの組を返す。
    tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}


if __name__ == "__main__":
    before, after = functions(sys.argv[1]), functions(sys.argv[2])
    for name in ("check", "predict"):
        print(f"{name}: {'変更なし' if before[name] == after[name] else '変更あり'}")
    print("足された関数:", "、".join(sorted(set(after) - set(before))) or "なし")
