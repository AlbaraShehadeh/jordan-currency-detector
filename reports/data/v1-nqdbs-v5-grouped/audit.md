# Data audit: `data/processed/v1-nqdbs-v5-grouped`

## Splits

| split | images | boxes | empty images |
|---|---:|---:|---:|
| train | 1727 | 4223 | 0 |
| valid | 407 | 825 | 0 |
| test | 386 | 870 | 0 |

## Class instances

| class | train | valid | test | total |
|---|---:|---:|---:|---:|
| 5-piastres | 506 | 99 | 105 | 710 |
| 10-piastres | 478 | 92 | 99 | 669 |
| quarter-dinar | 462 | 90 | 95 | 647 |
| half-dinar | 464 | 91 | 95 | 650 |
| 1-dinar | 479 | 94 | 98 | 671 |
| 5-dinar | 472 | 92 | 98 | 662 |
| 10-dinar | 433 | 85 | 89 | 607 |
| 20-dinar | 439 | 86 | 90 | 615 |
| 50-dinar | 490 | 96 | 101 | 687 |

## Leakage (train → eval)

- Grouping: Roboflow source id + flip-invariant pHash ≤ 6 bits
- Source groups: **953** (largest: 6 images)
- Groups spanning more than one split: **0**
- Eval (valid+test) images whose source group is also in train: **0 / 793 (0.0%)**
  - valid: 0
  - test: 0

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
