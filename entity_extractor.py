from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Canonical entity -> aliases. Keep this practical rather than exhaustive.
DEFAULT_ENTITY_LEXICON: Dict[str, List[str]] = {
    "person": [
        "person", "people", "man", "woman", "men", "women", "pedestrian", "pedestrians",
        "player", "players", "runner", "runners", "boy", "girl", "boys", "girls",
        "child", "children", "adult", "adults",
    ],
    "umbrella": ["umbrella", "umbrellas", "parasol"],
    "bag": ["bag", "bags", "backpack", "backpacks", "handbag", "handbags", "purse", "purses"],
    "phone": ["phone", "phones", "cell phone", "mobile phone", "smartphone"],
    "bottle": ["bottle", "bottles", "water bottle"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "ball": ["ball", "balls", "sports ball", "soccer ball", "basketball", "football", "tennis ball"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "car": ["car", "cars", "vehicle", "vehicles", "sedan", "sedans", "taxi", "taxis"],
    "bus": ["bus", "buses", "coach", "coaches"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "train": ["train", "trains"],
    "bench": ["bench", "benches", "seat", "seats"],
    "chair": ["chair", "chairs", "stool", "stools"],
    "table": ["table", "tables", "desk", "desks"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "horse": ["horse", "horses"],
    "bird": ["bird", "birds"],
    "tree": ["tree", "trees"],
    "plant": ["plant", "plants", "potted plant", "potted plants", "bush", "bushes"],
    "flower": ["flower", "flowers"],
    "building": ["building", "buildings", "house", "houses"],
    "window": ["window", "windows"],
    "door": ["door", "doors", "gate", "gates", "entrance", "entrances"],
    "fence": ["fence", "fences", "railing", "railings", "barrier", "barriers", "chain-link fence"],
    "sign": ["sign", "signs", "board", "boards", "notice", "notices"],
    "text": ["text", "texts", "word", "words", "letter", "letters", "writing", "written text"],
    "road": ["road", "roads", "street", "streets", "pavement", "lane", "lanes", "sidewalk", "sidewalks"],
    "crosswalk": ["crosswalk", "crosswalks", "zebra crossing", "zebra crossings", "pedestrian crossing", "pedestrian crossings"],
    "arrow": ["arrow", "arrows", "directional arrow", "road arrow", "painted arrow"],
    "traffic light": ["traffic light", "traffic lights", "signal light", "signal lights"],
    "stop sign": ["stop sign", "stop signs"],
    "field": ["field", "fields", "sports field", "sports fields", "court", "courts", "track", "tracks", "playground", "playgrounds"],
    "grass": ["grass", "lawn"],
    "sky": ["sky"],
    "cloud": ["cloud", "clouds"],
    "water": ["water", "river", "lake", "sea", "ocean"],
    "boat": ["boat", "boats", "ship", "ships"],
    "bridge": ["bridge", "bridges"],
    "pole": ["pole", "poles", "post", "posts"],
    "light": ["light", "lights", "lamp", "lamps", "streetlight", "streetlights"],
}

DEFAULT_STOPWORD_ENTITIES: Set[str] = {
    "image", "picture", "photo", "scene", "view", "thing", "things", "something", "someone",
    "somebody", "detail", "details", "area", "environment", "setting", "foreground", "background",
    "user", "assistant", "describe", "mention", "guess", "only", "clearly", "faithfully",
    "middle", "center", "centre", "left", "right", "front", "back", "top", "bottom", "kind",
    "sort", "type", "piece", "part", "parts", "side", "distance", "motion", "activity", "action",
}

FALLBACK_BLACKLIST = DEFAULT_STOPWORD_ENTITIES | {
    "many", "several", "multiple", "other", "another", "same", "new", "visible", "clear",
    "unclear", "likely", "possible", "possibly", "probably", "various", "different", "few",
    "group", "groups", "set", "sets", "lot", "lots",
}

ARTICLE_RE = r"(?:a|an|the|one|two|three|four|five|six|seven|eight|nine|ten|several|many|multiple)"

BAD_INTERNAL_TOKENS = {
    "is", "are", "was", "were", "be", "being", "been", "am",
    "on", "in", "at", "of", "from", "to", "with", "by", "and", "or", "than",
    "them", "their", "his", "her", "its", "our", "your", "my",
    "holding", "walking", "talking", "running", "standing", "sitting", "separates",
    "reads", "pointing", "wearing",
}
ADJ_HINTS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey", "orange",
    "pink", "purple", "small", "large", "big", "young", "old", "wooden", "metal", "painted",
    "chain-link", "chainlink", "sports", "traffic", "road", "street", "mobile", "cell", "potted",
}


@dataclass
class EntityMention:
    canonical: str
    mention: str
    sentence_idx: int
    span_start: int
    span_end: int
    source: str  # lexicon | fallback


@dataclass
class ExtractionResult:
    text: str
    sentences: List[str]
    mentions: List[EntityMention]
    raw_mentions: List[str]
    normalized_entities: List[str]
    sentence_entities: Dict[str, List[str]]
    used_lexicon_path: Optional[str] = None
    used_stopwords_path: Optional[str] = None


# -----------------------------------------------------------------------------
# Lexicon helpers
# -----------------------------------------------------------------------------

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_match(text: str) -> str:
    return _normalize_space(text.lower().replace("_", " ").replace("/", " "))


def load_entity_lexicon(path: Optional[str] = None) -> Dict[str, List[str]]:
    lexicon = {k: list(v) for k, v in DEFAULT_ENTITY_LEXICON.items()}
    if not path:
        return lexicon
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Entity lexicon JSON must be an object: canonical -> [aliases]")
    for canonical, aliases in obj.items():
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            raise ValueError(f"Invalid aliases for {canonical!r}")
        merged = set(lexicon.get(canonical, []))
        merged.update(str(x) for x in aliases)
        lexicon[str(canonical)] = sorted(merged)
    return lexicon


def load_stopword_entities(path: Optional[str] = None) -> Set[str]:
    stopwords = set(DEFAULT_STOPWORD_ENTITIES)
    if not path:
        return stopwords
    ext = Path(path).suffix.lower()
    text = Path(path).read_text(encoding="utf-8")
    if ext == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            stopwords.update(str(x).strip().lower() for x in obj if str(x).strip())
        else:
            raise ValueError("Stopword JSON must be a list of strings")
    else:
        for line in text.splitlines():
            s = line.strip().lower()
            if s and not s.startswith("#"):
                stopwords.add(s)
    return stopwords


def build_alias_to_canonical(lexicon: Dict[str, List[str]]) -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for canonical, aliases in lexicon.items():
        all_aliases = {_normalize_for_match(canonical)}
        all_aliases.update(_normalize_for_match(a) for a in aliases)
        for alias in all_aliases:
            if alias:
                alias_to_canonical[alias] = canonical
    return alias_to_canonical


# -----------------------------------------------------------------------------
# Text processing
# -----------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    text = _normalize_space(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]


def singularize_basic(word: str) -> str:
    word = word.lower().strip()
    irregular = {
        "people": "person",
        "men": "man",
        "women": "woman",
        "children": "child",
        "mice": "mouse",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_entity_phrase(
    phrase: str,
    alias_to_canonical: Dict[str, str],
    stopwords: Set[str],
) -> Optional[str]:
    phrase_norm = _normalize_for_match(phrase)
    if not phrase_norm:
        return None

    if phrase_norm in alias_to_canonical:
        return alias_to_canonical[phrase_norm]

    tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]*", phrase_norm) if t]
    if not tokens:
        return None

    # Trim determiners and descriptive words from the left.
    while tokens and (tokens[0] in stopwords or tokens[0] in ADJ_HINTS or re.fullmatch(ARTICLE_RE, tokens[0])):
        tokens.pop(0)
    if not tokens:
        return None

    # Try longest tail n-gram first.
    for n in range(min(3, len(tokens)), 0, -1):
        cand = " ".join(tokens[-n:])
        if cand in alias_to_canonical:
            return alias_to_canonical[cand]

    if any(tok in BAD_INTERNAL_TOKENS for tok in tokens):
        return None

    head = singularize_basic(tokens[-1])
    if head in stopwords or head in FALLBACK_BLACKLIST or len(head) <= 1:
        return None
    return head


def _compile_alias_patterns(lexicon: Dict[str, List[str]]) -> List[Tuple[str, str, re.Pattern[str]]]:
    rows: List[Tuple[str, str, re.Pattern[str]]] = []
    seen: Set[Tuple[str, str]] = set()
    for canonical, aliases in lexicon.items():
        full_aliases = {canonical, *aliases}
        for alias in full_aliases:
            alias_norm = _normalize_for_match(alias)
            key = (canonical, alias_norm)
            if not alias_norm or key in seen:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<!\w){re.escape(alias_norm)}(?!\w)")
            rows.append((canonical, alias_norm, pattern))
    rows.sort(key=lambda x: (-len(x[1]), x[0], x[1]))
    return rows


def _extract_fallback_phrases(sentence: str) -> List[str]:
    sent = _normalize_for_match(sentence)
    phrases: List[str] = []

    # Common noun-phrase style chunks after articles or numbers.
    chunk_pattern = re.compile(
        rf"\b(?:{ARTICLE_RE})\s+((?:[a-zA-Z][a-zA-Z\-]*\s+){{0,2}}[a-zA-Z][a-zA-Z\-]*)\b"
    )
    for m in chunk_pattern.finditer(sent):
        phrases.append(m.group(1).strip())

    # Keep unique order from the chunk-based recall pass only.
    seen = set()
    uniq = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# -----------------------------------------------------------------------------
# Public extraction API
# -----------------------------------------------------------------------------

def extract_entities(
    text: str,
    lexicon_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    allow_fallback: bool = True,
) -> ExtractionResult:
    lexicon = load_entity_lexicon(lexicon_path)
    stopwords = load_stopword_entities(stopwords_path)
    alias_to_canonical = build_alias_to_canonical(lexicon)
    alias_patterns = _compile_alias_patterns(lexicon)

    sentences = split_sentences(text)
    mentions: List[EntityMention] = []
    seen: Set[Tuple[int, int, int, str]] = set()

    for sent_idx, sentence in enumerate(sentences):
        sent_norm = _normalize_for_match(sentence)

        # 1) High-precision lexicon matching.
        for canonical, alias_norm, pattern in alias_patterns:
            for m in pattern.finditer(sent_norm):
                key = (sent_idx, m.start(), m.end(), canonical)
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(
                    EntityMention(
                        canonical=canonical,
                        mention=alias_norm,
                        sentence_idx=sent_idx,
                        span_start=m.start(),
                        span_end=m.end(),
                        source="lexicon",
                    )
                )

        # 2) Recall-oriented fallback phrases.
        if allow_fallback:
            for phrase in _extract_fallback_phrases(sentence):
                canonical = normalize_entity_phrase(phrase, alias_to_canonical, stopwords)
                if not canonical:
                    continue
                phrase_norm = _normalize_for_match(phrase)
                span_start = sent_norm.find(phrase_norm)
                span_end = span_start + len(phrase_norm) if span_start >= 0 else -1
                key = (sent_idx, span_start, span_end, canonical)
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(
                    EntityMention(
                        canonical=canonical,
                        mention=phrase_norm,
                        sentence_idx=sent_idx,
                        span_start=span_start,
                        span_end=span_end,
                        source="fallback",
                    )
                )

    mentions.sort(key=lambda x: (x.sentence_idx, x.span_start, x.span_end, x.canonical))
    raw_mentions = [m.mention for m in mentions]

    normalized_entities: List[str] = []
    seen_entities: Set[str] = set()
    sentence_entities: Dict[str, List[str]] = {}
    for m in mentions:
        if m.canonical not in seen_entities:
            normalized_entities.append(m.canonical)
            seen_entities.add(m.canonical)
        key = str(m.sentence_idx)
        sentence_entities.setdefault(key, [])
        if m.canonical not in sentence_entities[key]:
            sentence_entities[key].append(m.canonical)

    return ExtractionResult(
        text=text,
        sentences=sentences,
        mentions=mentions,
        raw_mentions=raw_mentions,
        normalized_entities=normalized_entities,
        sentence_entities=sentence_entities,
        used_lexicon_path=str(Path(lexicon_path).resolve()) if lexicon_path else None,
        used_stopwords_path=str(Path(stopwords_path).resolve()) if stopwords_path else None,
    )


def result_to_dict(result: ExtractionResult) -> Dict[str, Any]:
    return {
        "text": result.text,
        "sentences": result.sentences,
        "mentions": [asdict(m) for m in result.mentions],
        "raw_mentions": result.raw_mentions,
        "normalized_entities": result.normalized_entities,
        "sentence_entities": result.sentence_entities,
        "used_lexicon_path": result.used_lexicon_path,
        "used_stopwords_path": result.used_stopwords_path,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract verifiable entity mentions from generated text")
    parser.add_argument("--text", default=None, help="Inline text to extract entities from")
    parser.add_argument("--text-file", default=None, help="UTF-8 text file")
    parser.add_argument("--lexicon", default=None, help="Optional JSON lexicon: canonical -> [aliases]")
    parser.add_argument("--stopwords", default=None, help="Optional txt/json stopword entity list")
    parser.add_argument("--no-fallback", action="store_true", help="Disable heuristic fallback extraction")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    raise ValueError("Provide --text or --text-file")


def main() -> None:
    args = parse_args()
    text = _load_text(args)
    result = extract_entities(
        text=text,
        lexicon_path=args.lexicon,
        stopwords_path=args.stopwords,
        allow_fallback=not args.no_fallback,
    )
    payload = result_to_dict(result)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote extraction to: {Path(args.output).resolve()}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
