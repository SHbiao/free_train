from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from entity_extractor import extract_entities, result_to_dict as extraction_to_dict
from open_vocab_verifier import OpenVocabVerifier, aggregate_verifications, verification_to_dict

SPECULATIVE_PATTERNS = [
    r"\bappears?\s+to\b",
    r"\bseems?\s+to\b",
    r"\blooks?\s+like\b",
    r"\blikely\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bmay\s+be\b",
    r"\bmight\s+be\b",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def duplicate_sentence_count(text: str) -> int:
    sentences = split_sentences(text)
    seen: Dict[str, int] = {}
    for sent in sentences:
        norm = re.sub(r"\W+", " ", sent.lower()).strip()
        if norm:
            seen[norm] = seen.get(norm, 0) + 1
    return sum(v - 1 for v in seen.values() if v > 1)


def count_speculative_phrases(text: str) -> int:
    lowered = text.lower()
    return sum(len(re.findall(pat, lowered)) for pat in SPECULATIVE_PATTERNS)


# -----------------------------------------------------------------------------
# Run discovery
# -----------------------------------------------------------------------------

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


def discover_controlled_groups(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("group_")])


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


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_sample(
    sample: SampleRecord,
    verifier: OpenVocabVerifier,
    lexicon_path: Optional[str],
    stopwords_path: Optional[str],
    max_prompts_per_entity: int,
) -> Dict[str, Any]:
    extraction = extract_entities(
        text=sample.text,
        lexicon_path=lexicon_path,
        stopwords_path=stopwords_path,
        allow_fallback=True,
    )
    entities = extraction.normalized_entities

    verifications = verifier.verify_images(
        image_paths=[str(p) for p in sample.images if p.exists()],
        entities=entities,
        lexicon_path=lexicon_path,
        max_prompts_per_entity=max_prompts_per_entity,
    ) if entities else []

    aggregate = aggregate_verifications(verifications) if verifications else {
        "verified_entities": [],
        "unverified_entities": [],
        "entity_hits": [],
        "entity_support_ratio": 0.0,
    }

    return {
        "sample_id": sample.sample_id,
        "source_type": sample.source_type,
        "source_path": str(sample.source_path),
        "images": [str(p) for p in sample.images],
        "text": sample.text,
        "sentence_count": sample.n_sentences,
        "token_count_approx": count_tokens_approx(sample.text),
        "char_count": len(sample.text),
        "stop_reason": sample.stop_reason,
        "elapsed_sec": sample.elapsed_sec,
        "extraction": extraction_to_dict(extraction),
        "verification": {
            "per_image": [verification_to_dict(v) for v in verifications],
            "aggregate": aggregate,
        },
        "quality_flags": {
            "speculative_phrase_count": count_speculative_phrases(sample.text),
            "duplicate_sentence_count": duplicate_sentence_count(sample.text),
            "empty_text": not bool(sample.text.strip()),
        },
    }


# -----------------------------------------------------------------------------
# Summary and CSV
# -----------------------------------------------------------------------------

def summarize(rows: Sequence[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    stop_counts: Dict[str, int] = {}
    for row in rows:
        stop = str(row.get("stop_reason", "unknown"))
        stop_counts[stop] = stop_counts.get(stop, 0) + 1

    avg_tokens = safe_mean([r.get("token_count_approx") for r in rows])
    avg_sentences = safe_mean([r.get("sentence_count") for r in rows])
    avg_elapsed = safe_mean([r.get("elapsed_sec") for r in rows])
    avg_extracted = safe_mean([len(r.get("extraction", {}).get("normalized_entities", [])) for r in rows])
    avg_verified = safe_mean([len(r.get("verification", {}).get("aggregate", {}).get("verified_entities", [])) for r in rows])
    avg_unverified = safe_mean([len(r.get("verification", {}).get("aggregate", {}).get("unverified_entities", [])) for r in rows])
    avg_support_ratio = safe_mean([r.get("verification", {}).get("aggregate", {}).get("entity_support_ratio") for r in rows])
    avg_speculative = safe_mean([r.get("quality_flags", {}).get("speculative_phrase_count") for r in rows])
    avg_duplicate = safe_mean([r.get("quality_flags", {}).get("duplicate_sentence_count") for r in rows])

    return {
        "run_dir": str(run_dir),
        "n_samples": len(rows),
        "avg_token_count_approx": avg_tokens,
        "avg_sentence_count": avg_sentences,
        "avg_elapsed_sec": avg_elapsed,
        "avg_extracted_entity_count": avg_extracted,
        "avg_verified_entity_count": avg_verified,
        "avg_unverified_entity_count": avg_unverified,
        "avg_entity_support_ratio": avg_support_ratio,
        "avg_speculative_phrase_count": avg_speculative,
        "avg_duplicate_sentence_count": avg_duplicate,
        "stop_reason_counts": stop_counts,
        "notes": {
            "entity_support_ratio": "verified_entities / extracted_entities, aggregated across all images of a sample.",
            "verified_entities": "Detector-supported entities under open-vocabulary YOLO-World prompts; this is a support signal, not absolute ground truth.",
        },
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    columns = [
        "sample_id",
        "source_type",
        "images",
        "sentence_count",
        "token_count_approx",
        "stop_reason",
        "elapsed_sec",
        "extracted_entities",
        "verified_entities",
        "unverified_entities",
        "entity_support_ratio",
        "speculative_phrase_count",
        "duplicate_sentence_count",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            agg = row.get("verification", {}).get("aggregate", {})
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_type": row["source_type"],
                    "images": " | ".join(row["images"]),
                    "sentence_count": row["sentence_count"],
                    "token_count_approx": row["token_count_approx"],
                    "stop_reason": row["stop_reason"],
                    "elapsed_sec": row.get("elapsed_sec"),
                    "extracted_entities": " | ".join(row.get("extraction", {}).get("normalized_entities", [])),
                    "verified_entities": " | ".join(agg.get("verified_entities", [])),
                    "unverified_entities": " | ".join(agg.get("unverified_entities", [])),
                    "entity_support_ratio": agg.get("entity_support_ratio"),
                    "speculative_phrase_count": row.get("quality_flags", {}).get("speculative_phrase_count"),
                    "duplicate_sentence_count": row.get("quality_flags", {}).get("duplicate_sentence_count"),
                    "text": row["text"],
                }
            )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM outputs with entity extraction + YOLO-World open-vocab verification")
    parser.add_argument("run_dir", help="Run directory under vlm_cache/runs or vlm_control_cache/runs")
    parser.add_argument("--project-root", default=".", help="Project root to add to sys.path")
    parser.add_argument("--yolo-repo", default="./ultralytics-main", help="Ultralytics repo root")
    parser.add_argument("--weights", "--yolo-weights", dest="weights", required=True, help="YOLO-World weights path")
    parser.add_argument("--lexicon", default=None, help="Optional entity lexicon JSON")
    parser.add_argument("--stopwords", default=None, help="Optional stopword entity file")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-prompts-per-entity", type=int, default=3)
    parser.add_argument("--output-subdir", default="evaluation_open_vocab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    samples = discover_samples(run_dir)
    verifier = OpenVocabVerifier(
        project_root=args.project_root,
        yolo_repo=args.yolo_repo,
        weights_path=args.weights,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )

    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        row = evaluate_sample(
            sample=sample,
            verifier=verifier,
            lexicon_path=args.lexicon,
            stopwords_path=args.stopwords,
            max_prompts_per_entity=args.max_prompts_per_entity,
        )
        rows.append(row)

    summary = summarize(rows, run_dir)
    write_jsonl(out_dir / "evaluation_open_vocab.jsonl", rows)
    write_json(out_dir / "evaluation_open_vocab_summary.json", summary)
    write_csv(out_dir / "evaluation_open_vocab_table.csv", rows)

    print(f"[DONE] evaluated {len(rows)} samples")
    print(f"[EVAL DIR] {out_dir}")
    print(f"[AVG ENTITY SUPPORT] {summary.get('avg_entity_support_ratio')}")
    print(f"[AVG TOKENS] {summary.get('avg_token_count_approx')}")
    print(f"[STOP COUNTS] {summary.get('stop_reason_counts')}")


if __name__ == "__main__":
    main()
