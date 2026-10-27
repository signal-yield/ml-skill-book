# 二つの広告の価格と面積を、同じ単位で比べる
price_a = 98_000_000
area_a = 70
price_b = 107_000_000
area_b = 72
unit_a = price_a / area_a
unit_b = price_b / area_b
difference = unit_b - unit_a
ratio = unit_b / unit_a
print(f"物件A：{unit_a:,.2f} 円/㎡、{unit_a / 10_000:.4f} 万円/㎡")
print(f"物件B：{unit_b:,.2f} 円/㎡、{unit_b / 10_000:.4f} 万円/㎡")
print(f"㎡単価の差（B−A）：{difference:,.2f} 円/㎡、{difference / 10_000:.4f} 万円/㎡")
print(f"㎡単価の比率（B÷A）：{ratio:.6f} 倍")
print(f"Aを基準にした差の割合：{(ratio - 1) * 100:.4f}%")
