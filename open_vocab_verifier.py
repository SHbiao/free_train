from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from entity_extractor import DEFAULT_ENTITY_LEXICON, load_entity_lexicon


@dataclass
class PromptClass:
    prompt: str
    canonical: str


@dataclass
class EntityHit:
    entity: str
    verified_by_detector: bool
    max_confidence: float
    num_boxes: int
    matched_prompts: List[str]
    boxes: List[List[float]]


@dataclass
class VerificationResult:
    image_path: str
    model_type: str
    model_weights: str
    prompts: List[PromptClass]
    raw_detections: List[Dict[str, Any]]
    verified_entities: List[str]
    unverified_entities: List[str]
    entity_hits: List[EntityHit]


class OpenVocabVerifierError(RuntimeError):
    pass


class OpenVocabVerifier:
    def __init__(
        self,
        project_root: str,
        yolo_repo: str,
        weights_path: str,
        conf_threshold: float = 0.05,
        imgsz: int = 640,
        max_det: int = 100,
        use_half: bool = False,
        verbose: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.yolo_repo = Path(yolo_repo).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.use_half = use_half
        self.verbose = verbose

        for p in [self.project_root, self.yolo_repo]:
            ps = str(p)
            if ps not in sys.path:
                sys.path.insert(0, ps)

        self.model_type: str = "unknown"
        self.model = self._load_model()

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLOWorld  # type: ignore

            model = YOLOWorld(str(self.weights_path))
            self.model_type = "YOLOWorld"
            return model
        except Exception:
            pass

        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(str(self.weights_path))
            if not hasattr(model, "set_classes"):
                raise OpenVocabVerifierError(
                    "Loaded ultralytics model does not expose set_classes(); it does not look like a YOLO-World-style model."
                )
            self.model_type = "YOLO"
            return model
        except Exception as exc:
            raise OpenVocabVerifierError(
                f"Failed to import a YOLO-World-capable model from ultralytics. Details: {exc}"
            ) from exc

    @staticmethod
    def _canonical_prompt_expansions(entity: str, lexicon: Dict[str, List[str]]) -> List[str]:
        expansions = [entity]
        expansions.extend(lexicon.get(entity, []))
        # Keep unique, stable order.
        seen: set[str] = set()
        result: List[str] = []
        for x in expansions:
            s = str(x).strip().lower()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def build_prompt_classes(
        self,
        entities: Sequence[str],
        lexicon_path: Optional[str] = None,
        max_prompts_per_entity: int = 3,
    ) -> List[PromptClass]:
        lexicon = load_entity_lexicon(lexicon_path)
        rows: List[PromptClass] = []
        for entity in entities:
            canonical = str(entity).strip().lower()
            if not canonical:
                continue
            prompts = self._canonical_prompt_expansions(canonical, lexicon)[: max(1, max_prompts_per_entity)]
            for p in prompts:
                rows.append(PromptClass(prompt=p, canonical=canonical))
        return rows

    def verify_image(
        self,
        image_path: str,
        entities: Sequence[str],
        lexicon_path: Optional[str] = None,
        max_prompts_per_entity: int = 3,
    ) -> VerificationResult:
        image_path_resolved = str(Path(image_path).resolve())
        prompt_classes = self.build_prompt_classes(
            entities=entities,
            lexicon_path=lexicon_path,
            max_prompts_per_entity=max_prompts_per_entity,
        )
        if not prompt_classes:
            return VerificationResult(
                image_path=image_path_resolved,
                model_type=self.model_type,
                model_weights=str(self.weights_path),
                prompts=[],
                raw_detections=[],
                verified_entities=[],
                unverified_entities=[],
                entity_hits=[],
            )

        prompts = [x.prompt for x in prompt_classes]
        self.model.set_classes(prompts)
        results = self.model.predict(
            source=image_path_resolved,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=self.verbose,
            half=self.use_half,
        )
        if not results:
            results = []
        res = results[0] if results else None

        per_entity: Dict[str, Dict[str, Any]] = {
            x.canonical: {"max_confidence": 0.0, "num_boxes": 0, "matched_prompts": set(), "boxes": []}
            for x in prompt_classes
        }
        raw_detections: List[Dict[str, Any]] = []

        if res is not None:
            boxes = getattr(res, "boxes", None)
            if boxes is not None and getattr(boxes, "cls", None) is not None:
                cls_ids = boxes.cls.tolist()
                confs = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(cls_ids)
                xyxy = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else [[0.0, 0.0, 0.0, 0.0]] * len(cls_ids)
                for cls_id, conf, box in zip(cls_ids, confs, xyxy):
                    idx = int(cls_id)
                    if not (0 <= idx < len(prompt_classes)):
                        continue
                    prompt_row = prompt_classes[idx]
                    ent_state = per_entity[prompt_row.canonical]
                    ent_state["num_boxes"] += 1
                    ent_state["max_confidence"] = max(ent_state["max_confidence"], float(conf))
                    ent_state["matched_prompts"].add(prompt_row.prompt)
                    ent_state["boxes"].append([float(v) for v in box])
                    raw_detections.append(
                        {
                            "prompt": prompt_row.prompt,
                            "canonical": prompt_row.canonical,
                            "confidence": round(float(conf), 4),
                            "box": [round(float(v), 2) for v in box],
                        }
                    )

        entity_hits: List[EntityHit] = []
        verified_entities: List[str] = []
        unverified_entities: List[str] = []
        for entity in sorted(per_entity.keys()):
            state = per_entity[entity]
            hit = EntityHit(
                entity=entity,
                verified_by_detector=bool(state["num_boxes"] > 0),
                max_confidence=round(float(state["max_confidence"]), 4),
                num_boxes=int(state["num_boxes"]),
                matched_prompts=sorted(state["matched_prompts"]),
                boxes=[[round(float(v), 2) for v in box] for box in state["boxes"]],
            )
            entity_hits.append(hit)
            if hit.verified_by_detector:
                verified_entities.append(entity)
            else:
                unverified_entities.append(entity)

        return VerificationResult(
            image_path=image_path_resolved,
            model_type=self.model_type,
            model_weights=str(self.weights_path),
            prompts=prompt_classes,
            raw_detections=raw_detections,
            verified_entities=verified_entities,
            unverified_entities=unverified_entities,
            entity_hits=entity_hits,
        )

    def verify_images(
        self,
        image_paths: Sequence[str],
        entities: Sequence[str],
        lexicon_path: Optional[str] = None,
        max_prompts_per_entity: int = 3,
    ) -> List[VerificationResult]:
        rows: List[VerificationResult] = []
        for image_path in image_paths:
            rows.append(
                self.verify_image(
                    image_path=image_path,
                    entities=entities,
                    lexicon_path=lexicon_path,
                    max_prompts_per_entity=max_prompts_per_entity,
                )
            )
        return rows


def verification_to_dict(result: VerificationResult) -> Dict[str, Any]:
    return {
        "image_path": result.image_path,
        "model_type": result.model_type,
        "model_weights": result.model_weights,
        "prompts": [asdict(x) for x in result.prompts],
        "raw_detections": result.raw_detections,
        "verified_entities": result.verified_entities,
        "unverified_entities": result.unverified_entities,
        "entity_hits": [asdict(x) for x in result.entity_hits],
    }


def aggregate_verifications(rows: Sequence[VerificationResult]) -> Dict[str, Any]:
    entity_state: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for hit in row.entity_hits:
            state = entity_state.setdefault(
                hit.entity,
                {"max_confidence": 0.0, "num_boxes": 0, "images_verified": 0, "matched_prompts": set()},
            )
            if hit.verified_by_detector:
                state["images_verified"] += 1
            state["max_confidence"] = max(state["max_confidence"], hit.max_confidence)
            state["num_boxes"] += hit.num_boxes
            state["matched_prompts"].update(hit.matched_prompts)

    verified_entities = sorted([k for k, v in entity_state.items() if v["num_boxes"] > 0])
    unverified_entities = sorted([k for k, v in entity_state.items() if v["num_boxes"] == 0])

    aggregate_hits = []
    for entity in sorted(entity_state.keys()):
        state = entity_state[entity]
        aggregate_hits.append(
            {
                "entity": entity,
                "verified_by_detector": bool(state["num_boxes"] > 0),
                "max_confidence": round(float(state["max_confidence"]), 4),
                "num_boxes": int(state["num_boxes"]),
                "images_verified": int(state["images_verified"]),
                "matched_prompts": sorted(state["matched_prompts"]),
            }
        )

    return {
        "verified_entities": verified_entities,
        "unverified_entities": unverified_entities,
        "entity_hits": aggregate_hits,
        "entity_support_ratio": round(len(verified_entities) / max(len(entity_state), 1), 4) if entity_state else 0.0,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify extracted entities with YOLO-World open-vocabulary detection")
    parser.add_argument("image", nargs="+", help="One or more image paths")
    parser.add_argument("--entities", nargs="*", default=None, help="Inline canonical entities to verify")
    parser.add_argument("--entities-file", default=None, help="JSON/TXT file with one entity per line or a JSON list")
    parser.add_argument("--project-root", default=".", help="Project root to add to sys.path")
    parser.add_argument("--yolo-repo", default="./ultralytics-main", help="Ultralytics repo root")
    parser.add_argument("--weights", "--yolo-weights", dest="weights", required=True, help="YOLO-World weights path")
    parser.add_argument("--lexicon", default=None, help="Optional entity lexicon JSON")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-prompts-per-entity", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def _load_entities(args: argparse.Namespace) -> List[str]:
    entities = [str(x).strip().lower() for x in (args.entities or []) if str(x).strip()]
    if args.entities_file:
        raw = Path(args.entities_file).read_text(encoding="utf-8")
        if Path(args.entities_file).suffix.lower() == ".json":
            obj = json.loads(raw)
            if not isinstance(obj, list):
                raise ValueError("entities-file JSON must be a list")
            entities.extend(str(x).strip().lower() for x in obj if str(x).strip())
        else:
            entities.extend(line.strip().lower() for line in raw.splitlines() if line.strip())
    # Stable unique order.
    seen = set()
    uniq = []
    for x in entities:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    if not uniq:
        raise ValueError("Provide --entities and/or --entities-file")
    return uniq


def main() -> None:
    args = parse_args()
    entities = _load_entities(args)
    verifier = OpenVocabVerifier(
        project_root=args.project_root,
        yolo_repo=args.yolo_repo,
        weights_path=args.weights,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )
    rows = verifier.verify_images(
        image_paths=args.image,
        entities=entities,
        lexicon_path=args.lexicon,
        max_prompts_per_entity=args.max_prompts_per_entity,
    )
    payload = {
        "images": [str(Path(x).resolve()) for x in args.image],
        "entities": entities,
        "verifications": [verification_to_dict(x) for x in rows],
        "aggregate": aggregate_verifications(rows),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote verification to: {Path(args.output).resolve()}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
