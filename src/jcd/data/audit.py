"""Dataset audit: label sanity, class balance, near-duplicates and train->eval leakage."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from jcd.catalog import Catalog
from jcd.data.dataset import SPLITS, LabelParseError, YoloBox, YoloDataset, parse_label_file
from jcd.data.grouping import Grouping, group_dataset

_EPS = 1e-6
MAX_LISTED_ISSUES = 50


class LabelIssue(BaseModel):
    image: str
    issue: str


class SplitStats(BaseModel):
    images: int
    boxes: int
    empty_images: int
    class_counts: dict[str, int]


class LeakageStats(BaseModel):
    method: str
    n_groups: int
    largest_group: int
    groups_spanning_splits: int
    eval_images: int
    eval_images_leaked: int  # valid/test images whose group also has train images
    leaked_by_split: dict[str, int]

    @property
    def leaked_fraction(self) -> float:
        return self.eval_images_leaked / self.eval_images if self.eval_images else 0.0


class AuditReport(BaseModel):
    root: str
    raw_class_names: list[str]
    class_order: list[str]
    class_name_mapping: dict[str, str]
    splits: dict[str, SplitStats]
    instances_per_image: dict[int, int]
    image_sizes: dict[str, int]
    n_label_issues: int
    label_issues: list[LabelIssue]
    leakage: LeakageStats

    def to_markdown(self) -> str:
        ids = self.class_order
        lines = [
            f"# Data audit: `{self.root}`",
            "",
            "## Splits",
            "",
            "| split | images | boxes | empty images |",
            "|---|---:|---:|---:|",
            *(
                f"| {s} | {st.images} | {st.boxes} | {st.empty_images} |"
                for s, st in self.splits.items()
            ),
            "",
            "## Class instances",
            "",
            "| class | " + " | ".join(self.splits) + " | total |",
            "|---|" + "---:|" * (len(self.splits) + 1),
        ]
        for cid in ids:
            row = [self.splits[s].class_counts.get(cid, 0) for s in self.splits]
            lines.append(f"| {cid} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
        lk = self.leakage
        lines += [
            "",
            "## Leakage (train → eval)",
            "",
            f"- Grouping: {lk.method}",
            f"- Source groups: **{lk.n_groups}** (largest: {lk.largest_group} images)",
            f"- Groups spanning more than one split: **{lk.groups_spanning_splits}**",
            f"- Eval (valid+test) images whose source group is also in train: "
            f"**{lk.eval_images_leaked} / {lk.eval_images} ({lk.leaked_fraction:.1%})**",
            *(f"  - {s}: {n}" for s, n in lk.leaked_by_split.items()),
            "",
            "## Objects per image",
            "",
            "| objects | images |",
            "|---:|---:|",
            *(f"| {k} | {v} |" for k, v in sorted(self.instances_per_image.items())),
            "",
            "## Image sizes (top 5)",
            "",
            *(f"- {k}: {v}" for k, v in list(self.image_sizes.items())[:5]),
            "",
            f"## Label issues: {self.n_label_issues}",
            "",
            *(f"- `{i.image}`: {i.issue}" for i in self.label_issues),
        ]
        return "\n".join(lines).rstrip("\n") + "\n"


def _box_issues(box: YoloBox, n_classes: int) -> list[str]:
    issues = []
    if not 0 <= box.cls < n_classes:
        issues.append(f"class index {box.cls} out of range")
    if box.w <= 0 or box.h <= 0:
        issues.append("zero/negative box size")
    x1, y1 = box.xc - box.w / 2, box.yc - box.h / 2
    x2, y2 = box.xc + box.w / 2, box.yc + box.h / 2
    if min(x1, y1) < -_EPS or max(x2, y2) > 1 + _EPS:
        issues.append("box extends outside the image")
    return issues


def leakage_stats(splits: list[str], group_of: list[int], method: str) -> LeakageStats:
    members: dict[int, set[str]] = defaultdict(set)
    sizes: Counter[int] = Counter(group_of)
    for split, g in zip(splits, group_of, strict=True):
        members[g].add(split)
    leaked: Counter[str] = Counter()
    n_eval = 0
    for split, g in zip(splits, group_of, strict=True):
        if split != "train":
            n_eval += 1
            if "train" in members[g]:
                leaked[split] += 1
    return LeakageStats(
        method=method,
        n_groups=len(members),
        largest_group=max(sizes.values(), default=0),
        groups_spanning_splits=sum(1 for s in members.values() if len(s) > 1),
        eval_images=n_eval,
        eval_images_leaked=sum(leaked.values()),
        leaked_by_split={s: leaked[s] for s in SPLITS if s != "train"},
    )


def audit_dataset(
    ds: YoloDataset, catalog: Catalog, *, grouping: Grouping | None = None
) -> AuditReport:
    mapping = catalog.map_model_classes(ds.names)  # fail fast on unknown class names
    grouping = grouping or group_dataset(ds)

    issues: list[LabelIssue] = []
    split_counts: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    split_images: Counter[str] = Counter()
    split_empty: Counter[str] = Counter()
    per_image: Counter[int] = Counter()
    sizes: Counter[str] = Counter()

    for item in ds.items:
        split_images[item.split] += 1
        with Image.open(item.path) as im:
            sizes[f"{im.width}x{im.height}"] += 1
        try:
            boxes = parse_label_file(item.label_path)
        except LabelParseError as exc:
            issues.append(LabelIssue(image=item.path.name, issue=str(exc)))
            continue
        if not item.label_path.exists():
            issues.append(LabelIssue(image=item.path.name, issue="missing label file"))
        per_image[len(boxes)] += 1
        if not boxes:
            split_empty[item.split] += 1
        seen: set[tuple[int, float, float, float, float]] = set()
        for b in boxes:
            for msg in _box_issues(b, len(ds.names)):
                issues.append(LabelIssue(image=item.path.name, issue=msg))
            key = (b.cls, round(b.xc, 4), round(b.yc, 4), round(b.w, 4), round(b.h, 4))
            if key in seen:
                issues.append(LabelIssue(image=item.path.name, issue="duplicate box"))
            seen.add(key)
            if 0 <= b.cls < len(ds.names):
                split_counts[item.split][mapping[ds.names[b.cls]]] += 1

    present = [s for s in SPLITS if split_images[s]]
    return AuditReport(
        root=str(ds.root),
        raw_class_names=ds.names,
        class_order=list(catalog.ids),
        class_name_mapping=mapping,
        splits={
            s: SplitStats(
                images=split_images[s],
                boxes=sum(split_counts[s].values()),
                empty_images=split_empty[s],
                class_counts={c: split_counts[s][c] for c in catalog.ids if split_counts[s][c]},
            )
            for s in present
        },
        instances_per_image=dict(sorted(per_image.items())),
        image_sizes=dict(sizes.most_common()),
        n_label_issues=len(issues),
        label_issues=issues[:MAX_LISTED_ISSUES],
        leakage=leakage_stats(
            [it.split for it in ds.items],
            grouping.group_of,
            f"Roboflow source id + flip-invariant pHash ≤ {grouping.threshold} bits",
        ),
    )


def write_report(report: AuditReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "audit.json", out_dir / "audit.md"
    js.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md.write_text(report.to_markdown(), encoding="utf-8")
    return js, md
