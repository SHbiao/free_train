from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from api_client import GenerateParams, QwenVLMClient, VLMClientError

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "by", "with", "from", "for",
    "and", "or", "but", "as", "it", "its", "their", "there", "here", "into", "over",
    "under", "near", "some", "several", "many", "much", "very", "more", "most", "can",
    "could", "may", "might", "would", "should", "appears", "seems", "likely", "visible",
}


class ControllerError(RuntimeError):
    pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def list_images(input_paths: Sequence[str], recursive: bool) -> List[Path]:
    images: List[Path] = []
    for item in input_paths:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            images.append(p.resolve())
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for sub in p.glob(pattern):
                if sub.is_file() and sub.suffix.lower() in VALID_EXTS:
                    images.append(sub.resolve())
        else:
            raise FileNotFoundError(f"Invalid image input: {p}")
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError("No valid images found")
    return images


def chunk_images(images: Sequence[Path], as_multi_image: bool) -> Iterable[List[Path]]:
    if as_multi_image:
        yield list(images)
    else:
        for img in images:
            yield [img]


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    raise ValueError("You must provide --prompt or --prompt-file")


def extract_assistant_answer(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    idx = lower.rfind("assistant")
    if idx >= 0:
        text = text[idx + len("assistant"):]
    text = text.strip()
    # Clean common chat wrappers.
    text = re.sub(r"^[:：\-\s]+", "", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+", text)
    result = [p.strip() for p in pieces if p.strip()]
    return result


def normalize_one_sentence(text: str) -> str:
    text = extract_assistant_answer(text)
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    first = split_sentences(text)
    if first:
        text = first[0].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_keywords(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


SPECULATIVE_PATTERNS = [
    r"\bappears?\b",
    r"\bseems?\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\blooks like\b",
]


def sentence_score(candidate: str, accepted: Sequence[str]) -> Tuple[float, Dict[str, Any]]:
    c = candidate.strip()
    lower = c.lower()
    feats: Dict[str, Any] = {
        "is_stop": lower == "stop",
        "length_chars": len(c),
        "keyword_count": 0,
        "novel_keywords": 0,
        "max_similarity": 0.0,
        "speculative_hits": 0,
    }
    if not c:
        return -999.0, feats
    if lower == "stop":
        return -50.0, feats

    cand_kw = tokenize_keywords(c)
    feats["keyword_count"] = len(set(cand_kw))

    accepted_text = " ".join(accepted)
    accepted_kw = tokenize_keywords(accepted_text)
    novel_kw = set(cand_kw) - set(accepted_kw)
    feats["novel_keywords"] = len(novel_kw)

    sims = [jaccard(cand_kw, tokenize_keywords(s)) for s in accepted] if accepted else [0.0]
    feats["max_similarity"] = max(sims) if sims else 0.0

    speculative_hits = 0
    for pat in SPECULATIVE_PATTERNS:
        if re.search(pat, lower):
            speculative_hits += 1
    feats["speculative_hits"] = speculative_hits

    score = 0.0
    score += min(feats["novel_keywords"], 8) * 1.4
    score += min(feats["keyword_count"], 10) * 0.4
    score += max(0, 1.0 - feats["max_similarity"]) * 1.2
    score -= speculative_hits * 1.0

    if len(c) < 12:
        score -= 1.5
    elif len(c) > 220:
        score -= 1.0

    if accepted and feats["max_similarity"] >= 0.85:
        score -= 2.0

    return score, feats


def should_stop(
    accepted: Sequence[str],
    best_sentence: str,
    best_score: float,
    best_feats: Dict[str, Any],
    round_idx: int,
    max_sentences: int,
    min_score: float,
    min_novel_keywords: int,
    no_new_patience: int,
) -> Tuple[bool, str]:
    if best_sentence.strip().lower() == "stop":
        return True, "model_stop"
    if round_idx >= max_sentences:
        return True, "max_sentences"
    if best_score < min_score:
        return True, "score_below_threshold"
    if best_feats.get("novel_keywords", 0) < min_novel_keywords and accepted:
        no_new_recent = 1
        if no_new_recent >= no_new_patience:
            return True, "low_novelty"
    return False, "continue"


def build_round_prompt(base_prompt: str, accepted: Sequence[str], round_idx: int) -> str:
    accepted_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(accepted)) if accepted else "(none)"
    return (
        f"{base_prompt.strip()}\n\n"
        "You are in controlled continuation mode.\n"
        f"Current round: {round_idx}.\n"
        "Previously accepted sentences:\n"
        f"{accepted_block}\n\n"
        "Instructions:\n"
        "1. Output exactly one next sentence only.\n"
        "2. Mention only clearly visible objects, attributes, actions, or relations.\n"
        "3. Do not repeat accepted content.\n"
        "4. Do not speculate or infer hidden intent.\n"
        "5. If no reliable new visual information can be added, output STOP.\n"
        "6. Output plain text only.\n"
    )


def make_round_cache_key(image_paths: Sequence[Path], round_prompt: str, params: GenerateParams) -> str:
    payload = {
        "images": [sha256_file(p) for p in image_paths],
        "prompt_sha256": sha256_text(round_prompt),
        "params": asdict(params),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def cached_generate(
    client: QwenVLMClient,
    cache_dir: Path,
    image_paths: Sequence[Path],
    prompt: str,
    params: GenerateParams,
    force: bool,
) -> Dict[str, Any]:
    key = make_round_cache_key(image_paths, prompt, params)
    item_dir = cache_dir / key
    ensure_dir(item_dir)

    request_record = {
        "images": [str(p) for p in image_paths],
        "prompt": prompt,
        "params": asdict(params),
    }
    (item_dir / "request.json").write_text(json.dumps(request_record, ensure_ascii=False, indent=2), encoding="utf-8")

    response_path = item_dir / "response.json"
    answer_path = item_dir / "answer.txt"

    if response_path.exists() and answer_path.exists() and not force:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        payload["cached"] = True
        return payload

    result = client.generate_from_files(prompt=prompt, image_paths=[str(p) for p in image_paths], params=params)
    answer = normalize_one_sentence(result.text)
    payload = {
        "ok": result.ok,
        "status_code": result.status_code,
        "elapsed_sec": result.elapsed_sec,
        "n_images": result.n_images,
        "text": result.text,
        "answer": answer,
        "raw": result.raw,
        "cached": False,
    }
    response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_text(answer_path, answer)
    return payload


def run_controlled_generation(
    client: QwenVLMClient,
    image_paths: Sequence[Path],
    base_prompt: str,
    params: GenerateParams,
    run_dir: Path,
    candidates_per_round: int,
    max_sentences: int,
    min_score: float,
    min_novel_keywords: int,
    no_new_patience: int,
    force: bool,
) -> Dict[str, Any]:
    round_cache = run_dir / "round_cache"
    ensure_dir(round_cache)
    rounds_dir = run_dir / "rounds"
    ensure_dir(rounds_dir)

    accepted: List[str] = []
    total_elapsed = 0.0
    stop_reason = "max_sentences"

    for round_idx in range(1, max_sentences + 1):
        round_prompt = build_round_prompt(base_prompt, accepted, round_idx)
        round_dir = rounds_dir / f"round_{round_idx:02d}"
        ensure_dir(round_dir)
        save_text(round_dir / "prompt.txt", round_prompt)

        candidates: List[Dict[str, Any]] = []
        for cand_idx in range(1, candidates_per_round + 1):
            payload = cached_generate(
                client=client,
                cache_dir=round_cache,
                image_paths=image_paths,
                prompt=round_prompt,
                params=params,
                force=force,
            )
            answer = payload.get("answer", "")
            score, feats = sentence_score(answer, accepted)
            candidate_row = {
                "candidate_idx": cand_idx,
                "answer": answer,
                "score": score,
                "features": feats,
                "elapsed_sec": payload.get("elapsed_sec"),
                "cached": payload.get("cached", False),
            }
            candidates.append(candidate_row)
            append_jsonl(round_dir / "candidates.jsonl", candidate_row)
            total_elapsed += float(payload.get("elapsed_sec") or 0.0)

        # Deduplicate after scoring so repeated API outputs do not dominate.
        unique_candidates: Dict[str, Dict[str, Any]] = {}
        for row in candidates:
            key = row["answer"].strip().lower()
            if key not in unique_candidates or row["score"] > unique_candidates[key]["score"]:
                unique_candidates[key] = row
        ranked = sorted(unique_candidates.values(), key=lambda x: x["score"], reverse=True)
        if not ranked:
            stop_reason = "no_candidates"
            break

        best = ranked[0]
        (round_dir / "best.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        save_text(round_dir / "best.txt", best["answer"])

        stop, stop_reason = should_stop(
            accepted=accepted,
            best_sentence=best["answer"],
            best_score=float(best["score"]),
            best_feats=best["features"],
            round_idx=round_idx,
            max_sentences=max_sentences,
            min_score=min_score,
            min_novel_keywords=min_novel_keywords,
            no_new_patience=no_new_patience,
        )

        round_summary = {
            "round_idx": round_idx,
            "best": best,
            "stop": stop,
            "stop_reason": stop_reason,
            "accepted_before": list(accepted),
        }
        (round_dir / "round_summary.json").write_text(
            json.dumps(round_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if stop:
            if best["answer"].strip() and best["answer"].strip().lower() != "stop" and stop_reason == "max_sentences":
                accepted.append(best["answer"].strip())
            break

        accepted.append(best["answer"].strip())

    final_text = " ".join(accepted).strip()
    result = {
        "images": [str(p) for p in image_paths],
        "n_images": len(image_paths),
        "accepted_sentences": accepted,
        "final_text": final_text,
        "n_sentences": len(accepted),
        "stop_reason": stop_reason,
        "total_elapsed_sec": round(total_elapsed, 4),
        "params": asdict(params),
        "controller": {
            "candidates_per_round": candidates_per_round,
            "max_sentences": max_sentences,
            "min_score": min_score,
            "min_novel_keywords": min_novel_keywords,
            "no_new_patience": no_new_patience,
        },
    }
    (run_dir / "final_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_text(run_dir / "final_output.txt", final_text)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled sentence-by-sentence VLM generation")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--api-base", default="http://127.0.0.1:8002", help="API base URL")
    parser.add_argument("--prompt", default=None, help="Inline base prompt")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 base prompt text file")
    parser.add_argument("--cache-root", default="./vlm_control_cache", help="Directory for logs and cache")
    parser.add_argument("--run-name", default=None, help="Optional run folder name")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image directories")
    parser.add_argument("--as-multi-image", action="store_true", help="Send all collected images as one request group")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-novel-keywords", type=int, default=1)
    parser.add_argument("--no-new-patience", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_prompt = load_prompt(args)
    images = list_images(args.inputs, recursive=args.recursive)

    cache_root = Path(args.cache_root).resolve()
    ensure_dir(cache_root)
    runs_root = cache_root / "runs"
    ensure_dir(runs_root)

    run_name = args.run_name or utc_now_str()
    run_dir = runs_root / run_name
    ensure_dir(run_dir)

    params = GenerateParams(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    client = QwenVLMClient(
        base_url=args.api_base,
        timeout=args.timeout,
        retries=args.retries,
    )

    health = client.health()
    (run_dir / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    save_text(run_dir / "base_prompt.txt", base_prompt)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "api_base": args.api_base,
                "inputs": [str(p) for p in images],
                "params": asdict(params),
                "controller": {
                    "candidates_per_round": args.candidates_per_round,
                    "max_sentences": args.max_sentences,
                    "min_score": args.min_score,
                    "min_novel_keywords": args.min_novel_keywords,
                    "no_new_patience": args.no_new_patience,
                },
                "as_multi_image": args.as_multi_image,
                "recursive": args.recursive,
                "force": args.force,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = run_dir / "summary.jsonl"

    for group_idx, image_group in enumerate(chunk_images(images, as_multi_image=args.as_multi_image), start=1):
        group_dir = run_dir / f"group_{group_idx:03d}"
        ensure_dir(group_dir)
        try:
            result = run_controlled_generation(
                client=client,
                image_paths=image_group,
                base_prompt=base_prompt,
                params=params,
                run_dir=group_dir,
                candidates_per_round=args.candidates_per_round,
                max_sentences=args.max_sentences,
                min_score=args.min_score,
                min_novel_keywords=args.min_novel_keywords,
                no_new_patience=args.no_new_patience,
                force=args.force,
            )
            row = {
                "group_idx": group_idx,
                "images": [str(p) for p in image_group],
                "ok": True,
                **result,
            }
        except (VLMClientError, ControllerError, FileNotFoundError, ValueError) as exc:
            row = {
                "group_idx": group_idx,
                "images": [str(p) for p in image_group],
                "ok": False,
                "error": str(exc),
            }

        append_jsonl(summary_path, row)

    print(f"[DONE] controlled run saved to: {run_dir}")


if __name__ == "__main__":
    main()
