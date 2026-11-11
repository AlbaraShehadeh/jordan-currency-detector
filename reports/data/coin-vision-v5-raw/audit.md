# Data audit: `data/raw/coin-vision-v5`

## Splits

| split | images | boxes | empty images |
|---|---:|---:|---:|
| train | 840 | 1941 | 1 |
| valid | 129 | 308 | 0 |
| test | 37 | 127 | 0 |

## Class instances

| class | train | valid | test | total |
|---|---:|---:|---:|---:|
| 5-piastres | 222 | 40 | 19 | 281 |
| 10-piastres | 208 | 42 | 13 | 263 |
| quarter-dinar | 211 | 36 | 14 | 261 |
| half-dinar | 210 | 32 | 13 | 255 |
| 1-dinar | 217 | 32 | 16 | 265 |
| 5-dinar | 215 | 37 | 14 | 266 |
| 10-dinar | 198 | 30 | 16 | 244 |
| 20-dinar | 211 | 38 | 18 | 267 |
| 50-dinar | 249 | 21 | 4 | 274 |

## Leakage (train → eval)

- Grouping: Roboflow source id + flip-invariant pHash ≤ 6 bits
- Source groups: **995** (largest: 2 images)
- Groups spanning more than one split: **1**
- Eval (valid+test) images whose source group is also in train: **1 / 166 (0.6%)**
  - valid: 0
  - test: 1

## Objects per image

| objects | images |
|---:|---:|
| 0 | 1 |
| 1 | 503 |
| 2 | 133 |
| 3 | 137 |
| 4 | 123 |
| 5 | 46 |
| 6 | 26 |
| 7 | 11 |
| 8 | 5 |
| 9 | 18 |
| 10 | 1 |
| 14 | 1 |
| 15 | 1 |

## Image sizes (top 5)

- 3024x4032: 555
- 960x1280: 156
- 1200x1600: 94
- 4032x3024: 73
- 1600x1200: 36

## Label issues: 0
