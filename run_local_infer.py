from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from api_client import GenerateParams, QwenVLMClient, VLMClientError

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    raise ValueError("You must provide --prompt or --prompt-file")


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


def make_cache_key(image_paths: Sequence[Path], prompt: str, params: GenerateParams) -> str:
    image_hashes = [sha256_file(p) for p in image_paths]
    payload = {
        "images": image_hashes,
        "prompt_sha256": sha256_text(prompt),
        "params": asdict(params),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256_text(canonical)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_request_record(
    prompt: str,
    image_paths: Sequence[Path],
    params: GenerateParams,
    cache_key: str,
    mode: str,
) -> Dict[str, Any]:
    return {
        "cache_key": cache_key,
        "mode": mode,
        "prompt": prompt,
        "params": asdict(params),
        "images": [str(p) for p in image_paths],
        "image_sha256": {str(p): sha256_file(p) for p in image_paths},
    }


def call_and_cache(
    client: QwenVLMClient,
    prompt: str,
    image_paths: Sequence[Path],
    params: GenerateParams,
    cache_root: Path,
    run_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    cache_key = make_cache_key(image_paths, prompt, params)
    item_dir = cache_root / cache_key
    ensure_dir(item_dir)

    request_record = build_request_record(prompt, image_paths, params, cache_key, mode="multipart")
    save_text(item_dir / "prompt.txt", prompt)
    (item_dir / "request.json").write_text(
        json.dumps(request_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    response_json_path = item_dir / "response.json"
    output_txt_path = item_dir / "output.txt"

    if response_json_path.exists() and output_txt_path.exists() and not force:
        response = json.loads(response_json_path.read_text(encoding="utf-8"))
        response["cached"] = True
        return response

    result = client.generate_from_files(
        prompt=prompt,
        image_paths=[str(p) for p in image_paths],
        params=params,
    )
    response = {
        "ok": result.ok,
        "text": result.text,
        "n_images": result.n_images,
        "elapsed_sec": result.elapsed_sec,
        "status_code": result.status_code,
        "raw": result.raw,
        "cached": False,
    }
    response_json_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    save_text(output_txt_path, result.text)

    last_link_dir = run_dir / "items" / cache_key
    ensure_dir(last_link_dir.parent)
    if not last_link_dir.exists():
        try:
            os.symlink(item_dir, last_link_dir, target_is_directory=True)
        except Exception:
            # Windows / permission fallback: write a pointer file instead.
            last_link_dir.mkdir(parents=True, exist_ok=True)
            save_text(last_link_dir / "_pointer.txt", str(item_dir))

    return response


def chunk_images(images: Sequence[Path], as_multi_image: bool) -> Iterable[List[Path]]:
    if as_multi_image:
        yield list(images)
    else:
        for img in images:
            yield [img]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local experiment runner for the existing Qwen3-VL API")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--api-base", default="http://127.0.0.1:8002", help="API base URL")
    parser.add_argument("--prompt", default=None, help="Inline prompt")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 prompt text file")
    parser.add_argument("--cache-root", default="./vlm_cache", help="Directory for local cache and logs")
    parser.add_argument("--run-name", default=None, help="Optional run folder name")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image directories")
    parser.add_argument(
        "--as-multi-image",
        action="store_true",
        help="Send all collected images in a single multi-image request",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Ignore cache and call API again")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = load_prompt(args)
    images = list_images(args.inputs, recursive=args.recursive)

    cache_root = Path(args.cache_root).resolve()
    ensure_dir(cache_root)
    runs_root = cache_root / "runs"
    ensure_dir(runs_root)

    run_name = args.run_name or utc_now_str()
    run_dir = runs_root / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "items")

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

    health_info = client.health()
    (run_dir / "health.json").write_text(
        json.dumps(health_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_text(run_dir / "prompt.txt", prompt)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "api_base": args.api_base,
                "inputs": [str(p) for p in images],
                "params": asdict(params),
                "as_multi_image": args.as_multi_image,
                "recursive": args.recursive,
                "force": args.force,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_jsonl = run_dir / "summary.jsonl"
    summary_rows: List[Dict[str, Any]] = []

    for group_idx, image_group in enumerate(chunk_images(images, as_multi_image=args.as_multi_image), start=1):
        try:
            response = call_and_cache(
                client=client,
                prompt=prompt,
                image_paths=image_group,
                params=params,
                cache_root=cache_root / "cache_items",
                run_dir=run_dir,
                force=args.force,
            )
            row = {
                "group_idx": group_idx,
                "images": [str(p) for p in image_group],
                "n_images": len(image_group),
                "elapsed_sec": response.get("elapsed_sec"),
                "status_code": response.get("status_code"),
                "cached": response.get("cached", False),
                "text": response.get("text", ""),
            }
        except VLMClientError as exc:
            row = {
                "group_idx": group_idx,
                "images": [str(p) for p in image_group],
                "n_images": len(image_group),
                "error": str(exc),
            }

        append_jsonl(summary_jsonl, row)
        summary_rows.append(row)

    total = len(summary_rows)
    success = sum(1 for r in summary_rows if "error" not in r)
    print(f"[DONE] total groups = {total}, success = {success}, failed = {total - success}")
    print(f"[RUN DIR] {run_dir}")


if __name__ == "__main__":
    main()
