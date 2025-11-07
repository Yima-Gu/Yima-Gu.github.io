#!/usr/bin/env python3
"""Convert Obsidian-style PDF callout blocks into Fluid note directives.

Usage
-----
python convert_pdf_callouts.py FILE [FILE ...] [--no-backup]

For each file, the script looks for patterns like::

    > [!PDF|yellow] [[file.pdf#page=73&selection=...|My Title]]
    > > quoted line 1
    > > quoted line 2

and rewrites them to::

    {% note warning 'My Title (p.73)' %}
    quoted line 1
    quoted line 2
    {% endnote %}

The color name after `!PDF|` is mapped to a Fluid note style. Unsupported
colors fall back to ``info``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

CALL_OUT_RE = re.compile(
    r"^>\s*\[!PDF\|(?P<color>[^\]]+)\]\s*\[\[(?P<link>[^\]|]+)(?:\|(?P<title>.+))?\]\]\s*$"
)
QUOTE_LINE_RE = re.compile(r"^>\s*> ?(.*)$")

COLOR_MAP = {
    "yellow": "warning",
    "orange": "warning",
    "red": "danger",
    "green": "success",
    "blue": "info",
    "cyan": "info",
    "teal": "info",
    "purple": "primary",
    "grey": "default",
    "gray": "default",
}


def extract_page(link: str) -> str | None:
    match = re.search(r"#page=(\d+)", link)
    if match:
        return match.group(1)
    return None


def sanitise_title(title: str) -> str:
    escaped = title.replace("'", "\\'")
    return escaped.strip()


def build_note(color: str, title: str, body_lines: Iterable[str]) -> str:
    note_color = COLOR_MAP.get(color.lower().strip(), "info")
    body = "\n".join(line.rstrip() for line in body_lines).strip()
    if not body:
        body = "<em>(原引用为空)</em>"
    if title:
        header = sanitise_title(title)
        return f"{{% note {note_color} '{header}' %}}\n{body}\n{{% endnote %}}\n"
    return f"{{% note {note_color} %}}\n{body}\n{{% endnote %}}\n"


def transform_lines(lines: List[str]) -> List[str]:
    result: List[str] = []
    i = 0
    total = len(lines)
    while i < total:
        match = CALL_OUT_RE.match(lines[i])
        if match:
            color = match.group("color") or "info"
            link = match.group("link") or ""
            title = (match.group("title") or "").strip()
            page = extract_page(link)
            if page and title:
                if "p." not in title and "页" not in title:
                    title = f"{title}, p.{page}"
            elif page and not title:
                title = f"p.{page}"

            quote_lines: List[str] = []
            i += 1
            while i < total:
                line = lines[i]
                quote_match = QUOTE_LINE_RE.match(line)
                if quote_match:
                    quote_lines.append(quote_match.group(1))
                    i += 1
                    continue
                if line.strip() == ">":
                    i += 1
                    continue
                break

            result.append(build_note(color, title, quote_lines))
            continue

        result.append(lines[i])
        i += 1
    return result


def transform_text(text: str) -> str:
    lines = text.splitlines()
    transformed = transform_lines(lines)
    return "\n".join(transformed)


def process_file(path: Path, create_backup: bool) -> None:
    original = path.read_text(encoding="utf-8")
    converted = transform_text(original)
    if converted == original:
        print(f"No changes needed: {path}")
        return

    if create_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(original, encoding="utf-8")
            print(f"Created backup: {backup_path}")
    path.write_text(converted, encoding="utf-8")
    print(f"Updated: {path}")


def iterate_markdown_targets(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"Warning: {p} does not exist", file=sys.stderr)
            continue
        if p.is_dir():
            for md_file in p.rglob("*.md"):
                yield md_file
        elif p.suffix.lower() == ".md":
            yield p
        else:
            print(f"Skipping non-Markdown file: {p}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Obsidian-style PDF callouts into Hexo Fluid notes."
    )
    parser.add_argument("paths", nargs="+", help="Markdown files or directories to process.")
    parser.add_argument(
        "--no-backup",
        dest="create_backup",
        action="store_false",
        help="Do not create .bak backups before overwriting files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    any_updates = False
    for target in iterate_markdown_targets(args.paths):
        process_file(target, create_backup=args.create_backup)
        any_updates = True
    if not any_updates:
        print("No Markdown files were processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
