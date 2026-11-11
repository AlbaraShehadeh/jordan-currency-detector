# Data audit: `../Jordan-Coin-Detector-YOLOv8-based-Object-Detection-and-Counting/Jordan-Coins-Detection-4`

## Splits

| split | images | boxes | empty images |
|---|---:|---:|---:|
| train | 2069 | 4790 | 0 |
| valid | 237 | 599 | 0 |
| test | 214 | 529 | 0 |

## Class instances

| class | train | valid | test | total |
|---|---:|---:|---:|---:|
| 5-piastres | 576 | 74 | 60 | 710 |
| 10-piastres | 540 | 62 | 67 | 669 |
| quarter-dinar | 507 | 75 | 65 | 647 |
| half-dinar | 523 | 63 | 64 | 650 |
| 1-dinar | 539 | 75 | 57 | 671 |
| 5-dinar | 548 | 57 | 57 | 662 |
| 10-dinar | 492 | 64 | 51 | 607 |
| 20-dinar | 516 | 54 | 45 | 615 |
| 50-dinar | 549 | 75 | 63 | 687 |

## Leakage (train → eval)

- Grouping: Roboflow source id + flip-invariant pHash ≤ 6 bits
- Source groups: **953** (largest: 6 images)
- Groups spanning more than one split: **342**
- Eval (valid+test) images whose source group is also in train: **399 / 451 (88.5%)**
  - valid: 213
  - test: 186

## Objects per image

| objects | images |
|---:|---:|
| 1 | 1258 |
| 2 | 329 |
| 3 | 350 |
| 4 | 316 |
| 5 | 125 |
| 6 | 58 |
| 7 | 27 |
| 8 | 11 |
| 9 | 39 |
| 10 | 3 |
| 14 | 3 |
| 15 | 1 |

## Image sizes (top 5)

- 640x640: 2520

## Label issues: 0
