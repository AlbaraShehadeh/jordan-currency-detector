# Jordan Currency Detector

Detect, count and value **Jordanian coins and banknotes** (JOD) in images and video,
from a reproducible training pipeline to a deployed web app.

[![CI](https://github.com/AlbaraShehadeh/jordan-currency-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/AlbaraShehadeh/jordan-currency-detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)

> **Status: Phase 1 of 7: foundation.** The core domain logic (catalog, counting, video
> tracking registry) is implemented and tested. Inference, training, the API/UI and deployment
> land in later phases; see the [roadmap](#roadmap).

This is v2 of the
[Jordan Coin Detector](https://github.com/AlbaraShehadeh/Jordan-Coin-Detector-YOLOv8-based-Object-Detection-and-Counting)
research notebooks, rebuilt as an ML system. Design: [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).

## What it does

| Input | Output |
|---|---|
| Image | Boxes + denomination per item, per-denomination counts, **total value in JOD** |
| Video | Tracked items counted **once each** across frames, running total, annotated video |

Nine denominations: 5 and 10 piastres, ¼, ½ and 1 dinar (coins); 5, 10, 20 and 50 dinars (notes).

## Why a v2: what we found in v1

| Finding | Impact | v2 fix |
|---|---|---|
| **Train/test leakage**: the v1 dataset was an *augmented export* re-uploaded and re-split, so **399 of 451 val+test images (88 %) are augmented copies of training photos** | The reported mAP50 of 98.9 % is optimistic. Roboflow runs on a clean split scored ~91 % | Build from the 1,006 original photos and split by source photo |
| Image count used all boxes, value used only confident ones | Count and total could disagree | One filtered set drives both (`count_detections`) |
| Video locked each track's denomination to its first frame | One bad frame means the item is valued wrongly forever | Confidence-weighted vote over the track's lifetime, plus `min_hits` |
| Values keyed by class **index** | Reordering classes silently changes totals | Catalog keyed by **name** with aliases; mismatches fail at startup |
| Chained fine-tuning across 7 runs and 2 dataset versions | The model can't be reproduced | Config-driven training from pretrained weights with a fixed seed |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AlbaraShehadeh/jordan-currency-detector.git
cd jordan-currency-detector
make install            # uv sync + pre-commit hooks
make check              # lint + type-check + tests
uv run jcd catalog show
```

```python
from jcd.catalog import Catalog
from jcd.counting import count_detections
from jcd.types import Detection

catalog = Catalog.default()
dets = [
    Detection(box=(10, 10, 80, 80), label="1-2-dinar", confidence=0.93),
    Detection(box=(100, 10, 400, 200), label="20-dinar", confidence=0.88),
]
print(count_detections(dets, catalog, min_confidence=0.5).as_dict())
# {'currency': 'JOD', 'n_items': 2, 'counts': {'half-dinar': 1, '20-dinar': 1}, 'total': '20.50'}
```

## Project layout

```
src/jcd/          core library (framework-agnostic): catalog, counting, types, CLI
  resources/      bundled JOD denomination catalog
configs/          training / tracker configs (Phase 2+)
tests/            unit tests (integration tests land with the inference backends)
docs/             system design, ADRs, model card, eval reports
notebooks/        exploration only; pipelines run via the `jcd` CLI
```

## Roadmap

- [x] **0. Design**: [system design](docs/SYSTEM_DESIGN.md)
- [x] **1. Foundation**: repo, tooling, CI, core catalog and counting logic with tests
- [~] **2. Data**: ✅ audit + leakage-free grouped split ([v1 audit](reports/data/v1-coinsvision-nqdbs-v5/audit.md)) · ⏳ pull coin-vision originals, version on HF Hub
- [ ] **3. Inference core**: detector protocol, ONNX and Ultralytics backends, video pipeline
- [ ] **4. Training & evaluation**: reproducible training, counting/value metrics, model card
- [ ] **5. Serving**: FastAPI + Gradio UI
- [ ] **6. Ship**: Docker image and Hugging Face Space
- [ ] **7. Hardening**: metrics, load testing, in-the-wild test set

## Data & attribution

Images: [Coin Vision](https://universe.roboflow.com/jordancoinvision/coin-vision) by the
Roboflow user *jordancoinvision*, licensed **CC BY 4.0**.

## License

[AGPL-3.0](LICENSE). This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is AGPL-3.0.
