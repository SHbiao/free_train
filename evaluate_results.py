from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ------------------------------------------------------------
# Basic text utils
# ------------------------------------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECULATIVE_PATTERNS = [
    r"\bappears?\b",
    r"\bseems?\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\blooks like\b",
    r"\bseemingly\b",
]

GENERIC_ALIASES: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "woman", "boy", "girl", "pedestrian"],
    "umbrella": ["umbrella", "parasol"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike"],
    "car": ["car", "vehicle", "sedan", "taxi"],
    "bus": ["bus", "coach"],
    "truck": ["truck", "lorry"],
    "bench": ["bench", "seat"],
    "chair": ["chair", "stool"],
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "horse": ["horse"],
    "bird": ["bird"],
    "sports ball": ["ball", "sports ball", "basketball", "football", "soccer ball"],
    "traffic light": ["traffic light", "signal light"],
    "stop sign": ["stop sign", "road sign", "sign"],
    "potted plant": ["plant", "potted plant"],
    "cell phone": ["cell phone", "phone", "mobile phone"],
    # useful scene words for qualitative support, even if detector may miss them
    "road": ["road", "street", "pavement"],
    "crosswalk": ["crosswalk", "pedestrian crossing", "zebra crossing"],
    "fence": ["fence", "chain-link fence"],
    "field": ["field", "sports field", "court", "track"],
    "arrow": ["arrow"],
    "text": ["text", "word", "words", "letters", "sign text"],
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]


def count_tokens_approx(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def extract_assistant_answer(text: str) -> str:
    if not text:
        return ""
    norm = text.replace("\r\n", "\n")
    matches = list(re.finditer(r"(?:^|\n)assistant\s*\n", norm, flags=re.IGNORECASE))
    if matches:
        return norm[matches[-1].end():].strip()
    return norm.strip()


def text_contains_alias(text: str, alias: str) -> bool:
    t = text.lower()
    a = alias.lower().strip()
    if not a:
        return False
    if " " in a or "-" in a:
        return a in t
    return re.search(rf"\b{re.escape(a)}\b", t) is not None


def build_alias_table(yolo_names: Sequence[str]) -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = {}
    for name in yolo_names:
        key = str(name).lower().strip()
        aliases = {key}
        aliases.add(key.replace("_", " "))
        aliases.add(key.replace("-", " "))
        if key.endswith("s"):
            aliases.add(key[:-1])
        if key == "person":
            aliases.update(GENERIC_ALIASES.get("person", []))
        if key == "sports ball":
            aliases.update(GENERIC_ALIASES.get("sports ball", []))
        aliases.update(GENERIC_ALIASES.get(key, []))
        table[key] = sorted(a for a in aliases if a)

    for k, vals in GENERIC_ALIASES.items():
        table.setdefault(k, sorted(set(vals + [k])))
    return table


# ------------------------------------------------------------
# Optional backends: YOLO / CLIP
# ------------------------------------------------------------

@dataclass
class DetectorBundle:
    model: Any
    names: List[str]
    alias_table: Dict[str, List[str]]


class BackendError(RuntimeError):
    pass


def import_yolo_model(project_root: Path, yolo_repo_dir: Path, weights_path: Path) -> DetectorBundle:
    for p in [project_root, yolo_repo_dir]:
        ps = str(p.resolve())
        if ps not in sys.path:
            sys.path.insert(0, ps)
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise BackendError(
            f"Failed to import ultralytics. Check {yolo_repo_dir} or your environment. Details: {exc}"
        ) from exc

    model = YOLO(str(weights_path))
    raw_names = model.names
    if isinstance(raw_names, dict):
        names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
    else:
        names = [str(x) for x in raw_names]
    alias_table = build_alias_table(names)
    return DetectorBundle(model=model, names=names, alias_table=alias_table)


@dataclass
class ClipBundle:
    module: Any
    model: Any
    preprocess: Any
    device: str


def import_clip_bundle(project_root: Path, clip_repo_dir: Path, device: str, model_name: str, model_path: Optional[str]) -> ClipBundle:
    for p in [project_root, clip_repo_dir]:
        ps = str(p.resolve())
        if ps not in sys.path:
            sys.path.insert(0, ps)
    try:
        import torch  # type: ignore
        import clip  # type: ignore
    except Exception as exc:
        raise BackendError(f"Failed to import CLIP from {clip_repo_dir}. Details: {exc}") from exc

    load_target = model_path if model_path else model_name
    model, preprocess = clip.load(load_target, device=device)
    model.eval()
    return ClipBundle(module=clip, model=model, preprocess=preprocess, device=device)


# ------------------------------------------------------------
# Detection and CLIP scoring
# ------------------------------------------------------------


def run_yolo_on_image(detector: DetectorBundle, image_path: Path, conf: float, imgsz: int) -> Dict[str, Any]:
    results = detector.model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return {"image": str(image_path), "labels": [], "counts": {}, "boxes": 0}

    res = results[0]
    counts: Dict[str, int] = {}
    labels: List[str] = []

    boxes = getattr(res, "boxes", None)
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        cls_ids = boxes.cls.tolist()
        for cls_id in cls_ids:
            idx = int(cls_id)
            if 0 <= idx < len(detector.names):
                name = detector.names[idx].lower()
            else:
                name = str(idx)
            counts[name] = counts.get(name, 0) + 1
            labels.append(name)

    labels = sorted(set(labels))
    return {
        "image": str(image_path),
        "labels": labels,
        "counts": counts,
        "boxes": int(sum(counts.values())),
    }


def try_clip_score(clip_bundle: Optional[ClipBundle], image_path: Path, text: str) -> Optional[float]:
    if clip_bundle is None:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = clip_bundle.preprocess(image).unsqueeze(0).to(clip_bundle.device)
        text_tokens = clip_bundle.module.tokenize([text]).to(clip_bundle.device)
        with torch.no_grad():
            image_features = clip_bundle.model.encode_image(image_tensor)
            text_features = clip_bundle.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = float((image_features @ text_features.T).item())
        return score
    except Exception:
        return None


# ------------------------------------------------------------
# Run discovery
# ------------------------------------------------------------


def discover_controlled_groups(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("group_")])


@dataclass
class SampleRecord:
    sample_id: str
    images: List[Path]
    text: str
    n_sentences: int
    stop_reason: str
    elapsed_sec: Optional[float]
    source_type: str
    source_path: Path


def load_controlled_samples(run_dir: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for group_dir in discover_controlled_groups(run_dir):
        result_path = group_dir / "final_result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        images = [Path(p) for p in result.get("images", [])]
        text = str(result.get("final_text", "")).strip()
        records.append(
            SampleRecord(
                sample_id=group_dir.name,
                images=images,
                text=text,
                n_sentences=int(result.get("n_sentences", len(split_sentences(text)))),
                stop_reason=str(result.get("stop_reason", "unknown")),
                elapsed_sec=(float(result["total_elapsed_sec"]) if result.get("total_elapsed_sec") is not None else None),
                source_type="controlled",
                source_path=group_dir,
            )
        )
    return records


def load_local_infer_samples(run_dir: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    rows = read_jsonl(run_dir / "summary.jsonl")
    for idx, row in enumerate(rows, start=1):
        if row.get("error"):
            continue
        images = [Path(p) for p in row.get("images", [])]
        text = extract_assistant_answer(str(row.get("text", "")))
        records.append(
            SampleRecord(
                sample_id=f"group_{idx:03d}",
                images=images,
                text=text,
                n_sentences=len(split_sentences(text)),
                stop_reason="oneshot",
                elapsed_sec=(float(row["elapsed_sec"]) if row.get("elapsed_sec") is not None else None),
                source_type="oneshot",
                source_path=run_dir,
            )
        )
    return records


def discover_samples(run_dir: Path) -> List[SampleRecord]:
    groups = discover_controlled_groups(run_dir)
    if groups:
        return load_controlled_samples(run_dir)
    return load_local_infer_samples(run_dir)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------


def collect_mentions(text: str, alias_table: Dict[str, List[str]]) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for canonical, aliases in alias_table.items():
        hits = [a for a in aliases if text_contains_alias(text, a)]
        if hits:
            found[canonical] = sorted(set(hits))
    return found


def evaluate_sample(
    sample: SampleRecord,
    detector: DetectorBundle,
    clip_bundle: Optional[ClipBundle],
    yolo_conf: float,
    yolo_imgsz: int,
) -> Dict[str, Any]:
    detection_rows = [run_yolo_on_image(detector, img, conf=yolo_conf, imgsz=yolo_imgsz) for img in sample.images if img.exists()]
    detected_labels = sorted({lab for row in detection_rows for lab in row["labels"]})

    alias_table = detector.alias_table
    mentions = collect_mentions(sample.text, alias_table)
    mention_labels = sorted(mentions.keys())

    supported_labels: List[str] = []
    unsupported_labels: List[str] = []
    detected_set = set(detected_labels)
    for label in mention_labels:
        if label in detected_set:
            supported_labels.append(label)
        else:
            unsupported_labels.append(label)

    speculative_hits = 0
    lowered = sample.text.lower()
    for pat in SPECULATIVE_PATTERNS:
        speculative_hits += len(re.findall(pat, lowered))

    sentences = split_sentences(sample.text)
    dup_norm: Dict[str, int] = {}
    for sent in sentences:
        norm = re.sub(r"\W+", " ", sent.lower()).strip()
        dup_norm[norm] = dup_norm.get(norm, 0) + 1
    duplicate_sentence_count = sum(v - 1 for v in dup_norm.values() if v > 1)

    clip_scores = [try_clip_score(clip_bundle, img, sample.text) for img in sample.images if img.exists()]
    clip_scores = [x for x in clip_scores if x is not None]
    clip_score_mean = round(sum(clip_scores) / len(clip_scores), 4) if clip_scores else None

    return {
        "sample_id": sample.sample_id,
        "source_type": sample.source_type,
        "source_path": str(sample.source_path),
        "images": [str(p) for p in sample.images],
        "text": sample.text,
        "n_sentences": sample.n_sentences,
        "token_count_approx": count_tokens_approx(sample.text),
        "char_count": len(sample.text),
        "stop_reason": sample.stop_reason,
        "elapsed_sec": sample.elapsed_sec,
        "yolo": {
            "images": detection_rows,
            "detected_labels": detected_labels,
            "n_detected_labels": len(detected_labels),
        },
        "mentions": {
            "detected_supported_labels": supported_labels,
            "detector_unsupported_labels": unsupported_labels,
            "all_mentioned_labels": mention_labels,
            "mention_hits": mentions,
            "n_supported": len(supported_labels),
            "n_unsupported": len(unsupported_labels),
            "support_ratio": round(len(supported_labels) / max(len(mention_labels), 1), 4),
        },
        "quality_flags": {
            "speculative_hits": speculative_hits,
            "duplicate_sentence_count": duplicate_sentence_count,
            "empty_text": not bool(sample.text.strip()),
        },
        "clip": {
            "enabled": clip_bundle is not None,
            "score_mean": clip_score_mean,
            "score_count": len(clip_scores),
        },
    }


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------


def safe_mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def summarize(rows: Sequence[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    stop_counts: Dict[str, int] = {}
    for row in rows:
        stop = row.get("stop_reason", "unknown")
        stop_counts[stop] = stop_counts.get(stop, 0) + 1

    avg_tokens = safe_mean([float(r["token_count_approx"]) for r in rows])
    avg_sentences = safe_mean([float(r["n_sentences"]) for r in rows])
    avg_elapsed = safe_mean([float(r["elapsed_sec"]) for r in rows if r.get("elapsed_sec") is not None])
    avg_support_ratio = safe_mean([float(r["mentions"]["support_ratio"]) for r in rows])
    avg_speculative = safe_mean([float(r["quality_flags"]["speculative_hits"]) for r in rows])
    avg_duplicate = safe_mean([float(r["quality_flags"]["duplicate_sentence_count"]) for r in rows])

    clip_scores: List[float] = []
    for r in rows:
        score = r.get("clip", {}).get("score_mean")
        if score is not None:
            clip_scores.append(float(score))

    return {
        "run_dir": str(run_dir),
        "n_samples": len(rows),
        "avg_token_count_approx": avg_tokens,
        "avg_sentence_count": avg_sentences,
        "avg_elapsed_sec": avg_elapsed,
        "avg_detector_support_ratio": avg_support_ratio,
        "avg_speculative_hits": avg_speculative,
        "avg_duplicate_sentence_count": avg_duplicate,
        "avg_clip_score": safe_mean(clip_scores) if clip_scores else None,
        "stop_reason_counts": stop_counts,
        "notes": {
            "detector_unsupported_labels": "These are detector-unsupported mentions, not definitive hallucinations.",
            "support_ratio": "Mention support ratio is computed against YOLO-detected labels and alias matching.",
        },
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    columns = [
        "sample_id",
        "source_type",
        "images",
        "n_sentences",
        "token_count_approx",
        "stop_reason",
        "elapsed_sec",
        "detected_labels",
        "all_mentioned_labels",
        "detected_supported_labels",
        "detector_unsupported_labels",
        "support_ratio",
        "speculative_hits",
        "duplicate_sentence_count",
        "clip_score_mean",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_type": row["source_type"],
                    "images": " | ".join(row["images"]),
                    "n_sentences": row["n_sentences"],
                    "token_count_approx": row["token_count_approx"],
                    "stop_reason": row["stop_reason"],
                    "elapsed_sec": row.get("elapsed_sec"),
                    "detected_labels": " | ".join(row["yolo"]["detected_labels"]),
                    "all_mentioned_labels": " | ".join(row["mentions"]["all_mentioned_labels"]),
                    "detected_supported_labels": " | ".join(row["mentions"]["detected_supported_labels"]),
                    "detector_unsupported_labels": " | ".join(row["mentions"]["detector_unsupported_labels"]),
                    "support_ratio": row["mentions"]["support_ratio"],
                    "speculative_hits": row["quality_flags"]["speculative_hits"],
                    "duplicate_sentence_count": row["quality_flags"]["duplicate_sentence_count"],
                    "clip_score_mean": row["clip"]["score_mean"],
                    "text": row["text"],
                }
            )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM run outputs with local YOLO and optional CLIP support.")
    parser.add_argument("run_dir", help="Run directory under vlm_control_cache/runs or vlm_cache/runs")
    parser.add_argument("--project-root", default=".", help="Project root, e.g. D:/all_code/free_train")
    parser.add_argument("--yolo-repo", default="./ultralytics-main", help="Local ultralytics repo directory")
    parser.add_argument("--yolo-weights", default="./weight/yolo26n.pt", help="YOLO weights path")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--enable-clip", action="store_true", help="Enable optional CLIP image-text scoring")
    parser.add_argument("--clip-repo", default="./CLIP-main", help="Local CLIP repo directory")
    parser.add_argument("--clip-device", default="cuda", help="cuda or cpu")
    parser.add_argument("--clip-model-name", default="ViT-B/32")
    parser.add_argument("--clip-model-path", default=None, help="Optional local CLIP weight path")
    parser.add_argument("--output-dir", default=None, help="Where to save evaluation files; default: <run_dir>/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    yolo_repo = (project_root / args.yolo_repo).resolve() if not Path(args.yolo_repo).is_absolute() else Path(args.yolo_repo).resolve()
    yolo_weights = (project_root / args.yolo_weights).resolve() if not Path(args.yolo_weights).is_absolute() else Path(args.yolo_weights).resolve()
    clip_repo = (project_root / args.clip_repo).resolve() if not Path(args.clip_repo).is_absolute() else Path(args.clip_repo).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not yolo_repo.exists():
        raise FileNotFoundError(f"YOLO repo not found: {yolo_repo}")
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    detector = import_yolo_model(project_root=project_root, yolo_repo_dir=yolo_repo, weights_path=yolo_weights)

    clip_bundle: Optional[ClipBundle] = None
    clip_note = None
    if args.enable_clip:
        try:
            clip_bundle = import_clip_bundle(
                project_root=project_root,
                clip_repo_dir=clip_repo,
                device=args.clip_device,
                model_name=args.clip_model_name,
                model_path=args.clip_model_path,
            )
            clip_note = "enabled"
        except BackendError as exc:
            clip_note = f"disabled_due_to_error: {exc}"
            clip_bundle = None
    else:
        clip_note = "disabled"

    samples = discover_samples(run_dir)
    if not samples:
        raise FileNotFoundError(f"No evaluable samples found under: {run_dir}")

    rows = [evaluate_sample(s, detector, clip_bundle, yolo_conf=args.yolo_conf, yolo_imgsz=args.yolo_imgsz) for s in samples]
    summary = summarize(rows, run_dir)
    summary["clip_status"] = clip_note
    summary["yolo_backend"] = {
        "repo": str(yolo_repo),
        "weights": str(yolo_weights),
        "conf": args.yolo_conf,
        "imgsz": args.yolo_imgsz,
    }

    write_jsonl(output_dir / "evaluation.jsonl", rows)
    write_json(output_dir / "evaluation_summary.json", summary)
    write_csv(output_dir / "evaluation_table.csv", rows)

    print(f"[DONE] evaluated {len(rows)} samples")
    print(f"[EVAL DIR] {output_dir}")
    print(f"[AVG SUPPORT] {summary['avg_detector_support_ratio']}")
    print(f"[AVG TOKENS] {summary['avg_token_count_approx']}")
    print(f"[STOP COUNTS] {summary['stop_reason_counts']}")


if __name__ == "__main__":
    main()
