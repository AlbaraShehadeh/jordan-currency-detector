# Data audit: `data/processed/coin-vision-v5-grouped`

## Splits

| split | images | boxes | empty images |
|---|---:|---:|---:|
| train | 680 | 1713 | 1 |
| valid | 162 | 332 | 0 |
| test | 164 | 331 | 0 |

## Class instances

| class | train | valid | test | total |
|---|---:|---:|---:|---:|
| 5-piastres | 203 | 39 | 39 | 281 |
| 10-piastres | 190 | 36 | 37 | 263 |
| quarter-dinar | 188 | 37 | 36 | 261 |
| half-dinar | 183 | 36 | 36 | 255 |
| 1-dinar | 192 | 36 | 37 | 265 |
| 5-dinar | 192 | 37 | 37 | 266 |
| 10-dinar | 174 | 36 | 34 | 244 |
| 20-dinar | 193 | 37 | 37 | 267 |
| 50-dinar | 198 | 38 | 38 | 274 |

## Leakage (train → eval)

- Grouping: Roboflow source id + flip-invariant pHash ≤ 6 bits
- Source groups: **995** (largest: 2 images)
- Groups spanning more than one split: **0**
- Eval (valid+test) images whose source group is also in train: **0 / 326 (0.0%)**
  - valid: 0
  - test: 0

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
