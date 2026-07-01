#!/usr/bin/env python3
"""Batch-optimize Hexo post front matter: description, sticky, image paths."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "source" / "_posts"

SERIES_INTRO = {
    "Network/Network-01-Application-Layer.md": 10,
    "CSAPP/01-Bits, Bytes and Integers.md": 9,
    "Deep Learning/DL Note-1 Intro.md": 8,
    "algorithm/Introduction to Algorithm-01-Sorting.md": 7,
    "Formal_Language&Automata/形式语言与自动机.md": 6,
    "talks/The Era of Experience.md": 5,
}

TAG_REPLACEMENTS = {
    "DeepLearning": "Deep Learning",
}


def split_front_matter(content: str):
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None, content
    return match.group(1), content[match.end() :]


def extract_description(body: str, max_len: int = 120) -> str:
    text = re.sub(r"\{%.*?%\}", "", body, flags=re.DOTALL)
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        if line.startswith("#"):
            line = re.sub(r"^#+\s*", "", line)
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\$+[^$]+\$+", "", line)
        line = line.strip()
        if len(line) < 12:
            continue
        if len(line) > max_len:
            return line[: max_len - 1].rstrip() + "…"
        return line
    return ""


def fix_img_tags(body: str, post_path: Path) -> str:
    asset_dir = post_path.parent / post_path.stem

    def replace_img(match: re.Match) -> str:
        src = match.group(1).strip()
        if asset_dir.is_dir() and not (asset_dir / src).exists():
            return match.group(0)
        return f"![]({src})"

    return re.sub(
        r'<img\s+src="([^"]+)"\s+alt="[^"]*"(?:\s+width="(\d+)")?\s*>',
        replace_img,
        body,
    )


def normalize_tags(fm: str) -> str:
    for old, new in TAG_REPLACEMENTS.items():
        fm = fm.replace(f"  - {old}\n", f"  - {new}\n")
    return fm


def upsert_field(fm: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*(?:\n(?!\\S).*)*", re.MULTILINE)
    line = f'{key}: "{value}"' if value else f"{key}:"
    if pattern.search(fm):
        return pattern.sub(line, fm, count=1)
    return fm.rstrip() + f"\n{line}\n"


def process_file(path: Path) -> bool:
    rel = str(path.relative_to(POSTS_DIR))
    original = path.read_text(encoding="utf-8")
    fm, body = split_front_matter(original)
    if fm is None:
        return False

    changed = False
    new_fm = fm
    new_body = body

    if "description:" not in fm:
        desc = extract_description(body)
        if desc:
            desc = desc.replace('"', "'")
            new_fm = upsert_field(new_fm, "description", desc)
            changed = True

    if rel in SERIES_INTRO and "sticky:" not in fm:
        new_fm = new_fm.rstrip() + f"\nsticky: {SERIES_INTRO[rel]}\n"
        changed = True

    normalized_fm = normalize_tags(new_fm)
    if normalized_fm != new_fm:
        new_fm = normalized_fm
        changed = True

    fixed_body = fix_img_tags(body, path)
    if fixed_body != body:
        new_body = fixed_body
        changed = True

    if changed:
        path.write_text(f"---\n{new_fm.rstrip()}\n---\n{new_body}", encoding="utf-8")
    return changed


def main():
    targets = sorted(POSTS_DIR.rglob("*.md"))
    targets = [p for p in targets if not p.name.endswith(".bak")]
    updated = sum(process_file(p) for p in targets)
    print(f"Updated {updated} / {len(targets)} posts")


if __name__ == "__main__":
    main()
