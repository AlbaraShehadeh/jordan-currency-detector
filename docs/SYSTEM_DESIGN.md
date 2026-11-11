# Jordan Currency Detector — System Design

> Status: **Draft v0.1** · Date: 2026-11-10 · Supersedes: `Jordan-Coin-Detector-YOLOv8-based-Object-Detection-and-Counting` (v1, research notebooks)

---

## 1. Problem statement

Given an image, video, or live camera stream containing Jordanian currency (coins **and** banknotes),
**detect each item, classify its denomination, count it, and compute the total value in JOD.**

v1 proved feasibility (val mAP50 ≈ 0.97 with YOLOv8l) but lives in notebooks: non-reproducible
training lineage (train7 was fine-tuned from train6 → … → unknown), hardcoded secrets, counting bugs,
no serving layer. v2 turns it into a reproducible, tested, deployable ML system.

### 1.1 Users & use cases
| User | Use case | Input | Latency need |
|---|---|---|---|
| Portfolio visitor / recruiter | Upload a photo, see detections + total | image | interactive (< 2 s end-to-end) |
| Developer | Call an HTTP API | image (multipart / URL) | < 500 ms server-side on CPU |
| Demo / research | Count money passing in a video | video file | offline batch, progress bar |
| (stretch) Accessibility / cashier assist | Live webcam counting | stream | ≥ 10 FPS |

### 1.2 Non-goals (v2)
- Counterfeit detection.
- Foreign currencies.
- Mobile/edge app (ONNX export keeps the door open).
- Multi-tenant auth, billing.

---

## 2. Requirements

### Functional
- **F1** Image inference → boxes, class, confidence, per-denomination counts, total JOD, annotated image.
- **F2** Video inference → tracked, de-duplicated counts across frames, annotated output video.
- **F3** Webcam/stream inference in the UI (stretch).
- **F4** REST API with versioned routes and an OpenAPI schema.
- **F5** Web UI (Gradio) for F1–F3.
- **F6** CLI for batch inference, training, evaluation, export.
- **F7** Reproducible training + evaluation pipeline driven by config files.

### Non-functional
| ID | Requirement | Target |
|---|---|---|
| N1 | CPU latency, single 640px image (HF Spaces free tier, 2 vCPU) | p50 ≤ 300 ms, p95 ≤ 600 ms |
| N2 | Container image size | ≤ 2.5 GB (CPU-only torch or onnxruntime-only runtime) |
| N3 | Reproducibility | Any released model rebuildable from `git tag` + dataset version + config |
| N4 | Security | No secrets in code/history; secret scanning in CI; upload size/type limits |
| N5 | Quality | Lint + type check + tests gate every PR; core logic ≥ 85 % coverage |
| N6 | Traceability | Every API response carries `model_version` |

---

## 3. Success metrics

Detection mAP alone is not the product metric — the user cares about **"is the total right?"**

### Offline (test split, held out, never used for model selection)
| Metric | Why | v2 release gate |
|---|---|---|
| mAP50-95 (overall + per class) | Standard detection quality | ≥ v1 baseline on same split |
| Per-class recall @ deployed conf | Missed note = wrong total | ≥ 0.95 every class |
| **Exact-count accuracy** (% images where every per-class count is correct) | Product metric | ≥ 0.90 |
| **Value MAE (JOD)** and **% images with exact total** | Product metric | exact total ≥ 0.90 |
| Video count error (on a small hand-labeled clip set) | Tracking quality | ≤ 1 item per clip |
| CPU latency (p50/p95) | N1 | see N1 |

### Online (deployed)
Request latency p50/p95, error rate, requests by endpoint, model version, confidence distribution
(cheap drift signal).

---

## 4. Data

### 4.1 Provenance (verified 2026-11-11 via the Roboflow API)

| Roboflow project | Created | Versions | Notes |
|---|---|---|---|
| `jordancoinvision/coin-vision` | 2024-02-12 | v2, v3: augmented ×3, Roboflow-trained, **val mAP 91.1 / 91.4**. **v5 "Real Data": 1,006 originals, no augmentation** (840/129/37) | **The original dataset.** Augmentation settings match the v1 README. Owned by the user's workspace. |
| `coinsvision/jordan-coins-detection-nqdbs` | 2024-02-18 | v1 (Roboflow-trained, mAP 98.0), v5 (= v1 `model.pt`, mAP 98.9) | 2,520 images = an **augmented export of coin-vision re-uploaded as raw and randomly re-split**. Workspace not owned by the user (key leaked; revocation pending). Forked to `jordancoinvision/jordan-coins-detection-nqdbs-whbfw` as a backup. |

**Leakage finding (v1):** 2,520 images come from only 966 source photos (759 have 3 copies).
**399 of 451 val+test images (88 %)** share a source photo with train. Same-source pairs are
near-duplicates (mean pixel diff 38 vs 66 for unrelated pairs, after allowing flips). The v1 metrics
(98.9 mAP50) therefore measure memorization. The ~91 % from coin-vision's clean-val runs is the
best existing estimate of real performance.

### 4.2 Plan

| Item | v1 state | v2 plan |
|---|---|---|
| Source | Leaky `coinsvision/...nqdbs` v5 | **`jordancoinvision/coin-vision` v5 "Real Data"** (1,006 originals) |
| Split | Random over augmented copies | **Grouped split by source photo** (and near-duplicate cluster), stratified by class, ~70/15/15. The original test split (37 images) is too small to trust. |
| Augmentation | Baked into the dataset (×3) | **Online only** (Ultralytics train-time aug). The dataset stays raw. |
| Preprocessing | Stretch to 640×640 (distorts coins into ellipses) | Export without resize where possible; the model letterboxes |
| License | CC BY 4.0 | Attribute in README + dataset card |
| Storage | Committed to git | **HF Hub dataset repo**, versioned by revision; `jcd data pull` + checksum |
| Secrets | API key hardcoded in notebook (leaked) | `ROBOFLOW_API_KEY` from env / `.env` (gitignored); gitleaks in pre-commit + CI |

### 4.3 Data audit tasks (Phase 2, before training)
1. pHash near-duplicate clustering within the 1,006 originals (burst shots from one session), so the
   split groups whole clusters.
2. Class balance + instances-per-image histogram (dense scenes matter for counting).
3. Label sanity: zero-area boxes, out-of-bounds, duplicate boxes.
4. Coins vs notes failure modes: coin faces (heads/tails) and notes (front/back, folded, partial).
5. Re-evaluate the v1 `model.pt` on the new test split for an honest baseline. Caveat: v1 saw most
   source photos in training, so even this is optimistic. The in-the-wild set (§12) is the true check.

### 4.4 Domain catalog (single source of truth) ✅ implemented
`src/jcd/resources/currency_jod.yaml`, loaded by `jcd.catalog.Catalog`:
- Canonical ids: `5-piastres`, `10-piastres`, `quarter-dinar`, `half-dinar`, `1-dinar`, `5-dinar`,
  `10-dinar`, `20-dinar`, `50-dinar`.
- Raw class names resolve **by name** through aliases plus normalization (case, `/`, `⁄` U+2044,
  `_`, whitespace). So both `1-2-dinar` (nqdbs) and `1⁄2-dinar` (coin-vision) map to `half-dinar`.
- `Catalog.map_model_classes()` runs when a model loads and fails fast, listing every unknown class.
- Money is summed with `Decimal` (values quoted in YAML), never floats.

---

## 5. Modeling

### 5.1 Experiments
- Train **from COCO-pretrained weights** (no chained fine-tuning), fixed seed, config-driven.
- Candidates: YOLOv8n/s/m (v1 family) and the newest stable Ultralytics family at Phase 2
  (YOLO11 n/s/m, or newer if released and stable).
- Baseline row: v1 `model.pt` evaluated on the same test split with the same eval code.
- Choose on the **accuracy-vs-CPU-latency Pareto front**, gated by §3. Expect an *s*-size model
  exported to ONNX to beat v1's *l*-size model on latency with similar accuracy.
- Tune the deployed confidence threshold on **val** to maximize exact-count accuracy (not F1).

### 5.2 Tracking
- **MLflow** (local file store, no account) logs params, metrics, plots, and artifacts.
- The Ultralytics MLflow integration plus our own counting/value metrics as extra logged metrics.

### 5.3 Model registry & promotion
- Released weights (`.pt` + `.onnx` + `model_card.md` + `metrics.json`) are pushed to a
  **Hugging Face Hub model repo**, one revision tag per release (`v2.0.0`).
- The app resolves `MODEL_REPO` + `MODEL_REVISION` at startup (cached).
  Rollback = change revision env var.
- Promotion is manual: `make evaluate` must pass the §3 gates, and the eval report is attached to the
  release.

### 5.4 Compute
Training runs on a GPU notebook (Colab/Kaggle) that **only calls the repo's CLI**
(`jcd train --config configs/train/yolo11s.yaml`), so notebook and local runs are identical.
Apple M2 (MPS) works for smoke tests and small runs.

---

## 6. Inference logic (fixes v1 bugs)

### Image
`detect → filter(conf ≥ τ) → counts by class → total value`. The count and the value use the **same
filtered set**. (v1 counted unfiltered `len(boxes)` but valued only the filtered ones.)

### Video
```
frame → detector → tracker (ByteTrack / BoT-SORT, configurable) → TrackRegistry
TrackRegistry per track_id:
  - class votes (confidence-weighted) over the track's lifetime
  - hits / age
  - a track is COUNTED once it has ≥ k hits (default 3), suppressing flicker
  - its denomination = majority vote at read time (not first-frame class, as in v1)
```
- FPS measured as a moving average of real processing time (v1 formula was wrong).
- Output video keeps the source FPS (v1 hardcoded 30).
- Video processing runs as a **job** (generator yielding progress) so the UI shows progress and the
  API doesn't time out. In v2 the API caps upload length; in v3 it's an async job queue.

---

## 7. Architecture

```
                        ┌──────────────────── Docker image (CPU) ────────────────────┐
  Browser ──HTTP──▶     │  FastAPI app (uvicorn)                                      │
                        │   ├── /ui        Gradio Blocks (mounted)                    │
  API client ─HTTP─▶    │   ├── /v1/detect/image   POST multipart → JSON (+img opt)   │
                        │   ├── /v1/detect/video   POST (size-capped) → JSON + mp4    │
                        │   ├── /v1/model          model_version, classes, threshold  │
                        │   ├── /health, /ready                                       │
                        │   └── /metrics           Prometheus (optional)              │
                        │            │                                                │
                        │            ▼                                                │
                        │   jcd core library (pure Python, no web deps)               │
                        │   ├── catalog     Denomination catalog, value math          │
                        │   ├── detector    Detector protocol                         │
                        │   │     ├── OnnxDetector (onnxruntime, default in prod)     │
                        │   │     └── UltralyticsDetector (training/dev parity)       │
                        │   ├── counting    ImageCounter, TrackRegistry               │
                        │   ├── video       frame pipeline + tracker adapter          │
                        │   └── render      annotation / overlay                      │
                        └────────────▲───────────────────────────────────────────────┘
                                     │ download weights at startup (pinned revision, cached)
                         Hugging Face Hub: model repo (weights, card, metrics)
                                           dataset repo (versioned data)

  Training (GPU notebook or local):  jcd data pull → jcd train → jcd evaluate → jcd export → jcd release
                                      └── MLflow tracking (mlruns/)
```

**Design principles**
- **Core library is framework-agnostic.** API, UI, CLI, and tests all call `jcd`. No business
  logic in route handlers or Gradio callbacks.
- **Detector behind a Protocol.** Swapping ONNX ↔ Ultralytics (or a future model) touches one module.
- **Config over code.** `pydantic-settings` for runtime (env vars), YAML for training/eval.
- **Same model artifact everywhere.** No "notebook weights" vs "app weights".

---

## 8. Repository layout

```
jordan-currency-detector/
├── src/jcd/                  # installable package
│   ├── catalog.py  detector/  counting.py  video.py  render.py
│   ├── training/   (train.py, evaluate.py, export.py, metrics.py)
│   ├── data/       (pull.py, audit.py)
│   ├── api/        (app.py, routes.py, schemas.py, deps.py)
│   ├── ui/         (gradio_app.py)
│   ├── settings.py
│   └── cli.py      # Typer: jcd {data,train,evaluate,export,release,predict,serve}
├── configs/        # data.yaml, train/*.yaml, tracker/*.yaml  (catalog ships in src/jcd/resources/)
├── tests/          # unit/ (catalog, counting, registry), integration/ (api, onnx), fixtures/
├── notebooks/      # exploration only; v1 notebooks archived here (secrets removed)
├── docs/           # SYSTEM_DESIGN.md, MODEL_CARD.md, ADRs/, eval reports
├── docker/         # Dockerfile (multi-stage, CPU), .dockerignore
├── .github/workflows/  # ci.yml, release.yml
├── pyproject.toml  uv.lock  Makefile  .pre-commit-config.yaml
├── .env.example    LICENSE  README.md
```
Gitignored: data, weights, `mlruns/`, `runs/`, outputs. A tiny fixture set (a few images + a
2-second clip) is committed for tests.

---

## 9. Tech stack

| Concern | Choice | Why / alternative rejected |
|---|---|---|
| Language | Python 3.11 | Broad wheel support for torch/onnxruntime/ultralytics |
| Env & deps | **uv** + `pyproject.toml` + lockfile; extras `[train]`, `[serve]`, `[dev]` | Fast, reproducible; slim serve image without training deps |
| Model | Ultralytics YOLO (v8 baseline vs latest) | Proven on this data; native tracking |
| Serving runtime | **ONNX Runtime (CPU)** | Smaller and faster than torch on CPU; torch-free prod image |
| Tracking | Ultralytics ByteTrack/BoT-SORT trackers (or `supervision` ByteTrack with ONNX path) | Decided in an ADR during Phase 3 |
| API | **FastAPI** + Pydantic v2 + uvicorn | Typed schemas, OpenAPI for free |
| UI | **Gradio** (mounted on FastAPI) | Best for image/video/webcam ML demos; native to HF Spaces |
| CLI | Typer | One entrypoint for all pipelines |
| Experiment tracking | MLflow (local) | Free, no account; W&B optional |
| Data/model registry | Hugging Face Hub (dataset + model repos) | Free, versioned, integrates with Spaces; simpler than DVC here |
| Quality | ruff (lint+format), mypy, pytest + coverage, pre-commit, gitleaks | Standard |
| Container | Docker multi-stage, non-root, CPU | Runs on Spaces / any host |
| CI/CD | GitHub Actions: PR → lint/type/test/build; tag → build & deploy to HF Space | |
| Deploy | **HF Spaces (Docker SDK)** + image runnable anywhere | Free public demo; portable |
| Observability | structlog JSON logs, request timing middleware, optional Prometheus `/metrics` | |

---

## 10. Security, licensing, compliance
- **Immediate:** rotate the Roboflow API key leaked in v1 `Training/yolov8_train.ipynb` (it's in
  public git history). The new repo starts with clean history.
- Secrets only through env vars / HF Space secrets. Gitleaks runs in pre-commit and CI.
- Uploads: max size (image 10 MB, video 50 MB / 60 s), MIME + decode validation, no files persisted
  beyond the request (temp dir cleanup).
- **License:** Ultralytics is **AGPL-3.0**, so this repo is released under AGPL-3.0. The dataset is
  CC BY 4.0, with attribution.

---

## 11. CI/CD

```
PR / push ──▶ ci.yml:  uv sync → ruff → mypy → pytest (unit + API with tiny ONNX fixture) → docker build
tag v*    ──▶ release.yml:  build image → smoke test /health + sample request → push to HF Space
Model release (manual, GPU):  jcd evaluate (gates) → jcd export → jcd release --tag vX.Y.Z → HF Hub
```
Code releases and model releases are decoupled. The Space pins `MODEL_REVISION`.

---

## 12. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Split leakage inflates v1's 0.97 mAP | pHash audit; re-split by session if needed; report honestly |
| Small/biased dataset (lighting, backgrounds) → real-world drop | Collect a small "in-the-wild" test set (phone photos); report separately |
| HF Spaces CPU too slow for video | ONNX + smaller model + frame stride option; video length cap |
| Coins vs notes scale differences (tiny coins in note-heavy scenes) | Evaluate per-size buckets; try imgsz 800/960 if recall on coins lags |
| Tracker ID switches double-count | min-hits + vote-based registry; measure on labeled clips |
| AGPL obligations | Repo is AGPL; source is public |

---

## 13. Roadmap

| Phase | Deliverables | Exit criteria |
|---|---|---|
| **0. Design** ✅ (this doc) | SYSTEM_DESIGN.md, decisions agreed | User sign-off |
| **1. Foundation** ✅ | Repo scaffold, uv/pyproject, ruff/mypy/pytest, pre-commit, CI skeleton, README, LICENSE, `jcd` catalog + counting core with tests | CI green; `jcd --help` works |
| **2. Data** | `jcd data pull/audit/split` from coin-vision v5, near-dup clustering, grouped split, dataset on HF Hub, audit report, v1 baseline re-eval | Data versioned; split leakage = 0 |
| **3. Inference core** | Detector protocol, Ultralytics + ONNX backends, image counter, TrackRegistry, renderer; v1 `model.pt` as interim model | Unit + golden-image tests pass; bugs from §6 fixed |
| **4. Training & eval** | `jcd train/evaluate/export/release`, MLflow, Colab runner notebook, experiment table, model card | New model passes §3 gates; published to HF Hub |
| **5. Serving** | FastAPI routes, Gradio UI (image/video/webcam), settings, logging, limits | API tests pass; latency measured vs N1 |
| **6. Ship** | Dockerfile, release workflow, HF Space live, README with demo GIF + badges | Public URL serving the pinned model |
| **7. Hardening** (optional) | Prometheus metrics, load test (locust), in-the-wild test set, ADRs | |

## 14. Open questions
1. ~~Repo/package name~~: `jordan-currency-detector` / `jcd` (adopted).
2. GitHub account for the new repo (`gh` is logged in as `albara-shehadeh-ai-crunch`; v1 lives under `AlbaraShehadeh`).
3. HF Hub username/org for the model, dataset, and Space.
4. Can you shoot ~30–50 new phone photos and 3–5 short clips for the in-the-wild test set?
5. Revoke the leaked `coinsvision` key (needs a `coinsvision` admin or Roboflow support).
6. Is the `1-dinar` class the coin only, or also the 1 JOD banknote? (Both exist; the value is the same.)
