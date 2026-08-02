#!/usr/bin/env python3
"""Build the dependency-free Riftco Transformer documentation site.

``site/docs_manifest.json`` is the source of truth for the reader-oriented
documentation hierarchy and page metadata.  The repository deliberately avoids
a Python Markdown dependency, so this script renders the subset used by
README.md and docs/*.md, safely rewrites cross-guide links, and emits a
byte-for-byte deterministic GitHub Pages site with canonical and legacy routes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_SOURCE = PROJECT_ROOT / "site"
DOCS_SOURCE = PROJECT_ROOT / "docs"
MANIFEST_PATH = SITE_SOURCE / "docs_manifest.json"
TEMPLATES_DIRECTORY = SITE_SOURCE / "templates"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "pages"
DEFAULT_BASE_URL = "https://quangng2000.github.io/riftco-transformer/"
REPOSITORY_BLOB_URL = "https://github.com/quangng2000/riftco-transformer/blob/main/"
REPOSITORY_EDIT_URL = "https://github.com/quangng2000/riftco-transformer/edit/main/"
REPOSITORY_ISSUES_URL = "https://github.com/quangng2000/riftco-transformer/issues/new"

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^`]*)$")
_LIST_RE = re.compile(r"^(?P<space>[ \t]*)(?P<marker>[-+*]|\d+[.)])[ \t]+(?P<body>.*)$")
_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_HORIZONTAL_RULE_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
_PLACEHOLDER_RE = re.compile(r"{{[A-Z0-9_]+}}")


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    identifier: str


@dataclass(frozen=True)
class Section:
    identifier: str
    title: str
    description: str
    order: int

    @property
    def canonical_path(self) -> str:
        return f"docs/{self.identifier}/index.html"


@dataclass(frozen=True)
class Guide:
    source: Path
    source_name: str
    slug: str
    title: str
    summary: str
    section: Section
    guide_type: str
    order: int
    stability: str
    backends: tuple[str, ...]
    audience: tuple[str, ...]
    related_sources: tuple[str, ...]
    content: str
    headings: tuple[Heading, ...]
    search_text: str

    @property
    def canonical_path(self) -> str:
        return f"docs/{self.section.identifier}/{self.slug}.html"

    @property
    def legacy_path(self) -> str:
        return f"guides/{self.slug}.html"

    @property
    def title_anchor(self) -> str:
        first_heading = self.headings[0] if self.headings else None
        if first_heading and first_heading.level == 1:
            return first_heading.identifier
        return _slugify(self.title)


@dataclass(frozen=True)
class Documentation:
    version: str
    sections: tuple[Section, ...]
    guides: tuple[Guide, ...]

    def guides_in(self, section: Section) -> tuple[Guide, ...]:
        return tuple(guide for guide in self.guides if guide.section == section)


def _slugify(value: str, fallback: str = "section") -> str:
    # Match the useful part of GitHub's heading-slug behavior: punctuation is
    # removed (so `F/P/T/I` becomes `fpti`) and whitespace becomes a dash.
    plain = re.sub(r"[`*~]", "", value).lower()
    plain = re.sub(r"[^a-z0-9_\s-]", "", plain)
    slug = re.sub(r"\s+", "-", plain).strip("-")
    return slug or fallback


def _guide_slug(path: Path) -> str:
    # Public guide URLs mirror the repository's lowercase filenames.  Keeping
    # underscores also makes links easy to predict from docs/TENSOR_OPS.md.
    slug = re.sub(r"[^a-z0-9_-]+", "-", path.stem.lower()).strip("-")
    return slug or "guide"


def _normalise_space(value: str) -> str:
    return " ".join(value.split())


def _html_to_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _normalise_space(html.unescape(without_tags))


def _indent_width(value: str) -> int:
    width = 0
    for character in value:
        width += 4 if character == "\t" else 1
    return width


def _split_destination(value: str) -> tuple[str, str | None]:
    value = value.strip()
    titled = re.match(r"^(\S+)[ \t]+(?:\"([^\"]*)\"|'([^']*)')$", value)
    if titled:
        return titled.group(1), titled.group(2) or titled.group(3) or ""
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1], None
    return value, None


class MarkdownRenderer:
    """Render one repository Markdown document to an escaped HTML fragment."""

    def __init__(self, source: Path, guide_urls: dict[Path, str]) -> None:
        self.source = source.resolve()
        self.guide_urls = guide_urls
        self.headings: list[Heading] = []
        self._heading_counts: dict[str, int] = {}

    def render(self, markdown: str) -> str:
        lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        rendered, _ = self._render_blocks(lines, 0, len(lines))
        return "\n".join(rendered)

    def _render_blocks(
        self,
        lines: list[str],
        start: int,
        stop: int,
    ) -> tuple[list[str], int]:
        output: list[str] = []
        index = start
        while index < stop:
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            fence = _FENCE_RE.match(line)
            if fence:
                block, index = self._render_fence(lines, index, stop, fence)
                output.append(block)
                continue

            heading = _HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                raw_title = heading.group(2).strip()
                title = _html_to_text(self.render_inline(raw_title))
                base_identifier = _slugify(title)
                count = self._heading_counts.get(base_identifier, 0) + 1
                self._heading_counts[base_identifier] = count
                identifier = base_identifier if count == 1 else f"{base_identifier}-{count}"
                self.headings.append(Heading(level, title, identifier))
                output.append(
                    f'<h{level} id="{identifier}">'
                    f"{self.render_inline(raw_title)}"
                    f'<a class="anchor-link" href="#{identifier}" '
                    f'aria-label="Link to {html.escape(title, quote=True)}">#</a>'
                    f"</h{level}>"
                )
                index += 1
                continue

            if self._is_table_start(lines, index, stop):
                table, index = self._render_table(lines, index, stop)
                output.append(table)
                continue

            if line.lstrip().startswith(">"):
                quoted: list[str] = []
                while index < stop:
                    match = re.match(r"^[ \t]*>[ \t]?(.*)$", lines[index])
                    if not match:
                        break
                    quoted.append(match.group(1))
                    index += 1
                inner, _ = self._render_blocks(quoted, 0, len(quoted))
                output.append("<blockquote>\n" + "\n".join(inner) + "\n</blockquote>")
                continue

            list_match = _LIST_RE.match(line)
            if list_match:
                list_html, index = self._render_list(
                    lines,
                    index,
                    stop,
                    _indent_width(list_match.group("space")),
                )
                output.append(list_html)
                continue

            if _HORIZONTAL_RULE_RE.match(line):
                output.append("<hr>")
                index += 1
                continue

            paragraph: list[str] = [line.strip()]
            index += 1
            while index < stop and lines[index].strip():
                if self._starts_block(lines, index, stop):
                    break
                paragraph.append(lines[index].strip())
                index += 1
            output.append(f"<p>{self.render_inline(' '.join(paragraph))}</p>")

        return output, index

    def _starts_block(self, lines: list[str], index: int, stop: int) -> bool:
        line = lines[index]
        return bool(
            _FENCE_RE.match(line)
            or _HEADING_RE.match(line)
            or _LIST_RE.match(line)
            or line.lstrip().startswith(">")
            or _HORIZONTAL_RULE_RE.match(line)
            or self._is_table_start(lines, index, stop)
        )

    def _render_fence(
        self,
        lines: list[str],
        index: int,
        stop: int,
        opening: re.Match[str],
    ) -> tuple[str, int]:
        marker = opening.group(1)
        language = opening.group(2).strip().split(maxsplit=1)[0] if opening.group(2).strip() else ""
        index += 1
        body: list[str] = []
        closing = re.compile(rf"^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
        while index < stop and not closing.match(lines[index]):
            body.append(lines[index])
            index += 1
        if index < stop:
            index += 1

        source = "\n".join(body)
        escaped = html.escape(source)
        safe_language = re.sub(r"[^a-zA-Z0-9_+.-]", "", language.lower())
        if safe_language == "mermaid":
            return (
                '<div class="diagram-block" data-diagram="mermaid">\n'
                f'<div class="mermaid">{escaped}</div>\n'
                "</div>",
                index,
            )
        if safe_language in {"math", "latex", "tex"}:
            return (
                '<div class="math-block" data-math="display">\\[\n'
                f"{escaped}\n"
                "\\]</div>",
                index,
            )

        language_class = f" language-{safe_language}" if safe_language else ""
        label = (
            f'<span class="code-language">{html.escape(safe_language)}</span>'
            if safe_language
            else ""
        )
        return (
            f'<pre class="code-block">{label}'
            f'<code class="{language_class.strip()}">{escaped}</code></pre>',
            index,
        )

    def _render_list(
        self,
        lines: list[str],
        index: int,
        stop: int,
        base_indent: int,
    ) -> tuple[str, int]:
        first = _LIST_RE.match(lines[index])
        assert first is not None
        ordered = first.group("marker")[0].isdigit()
        tag = "ol" if ordered else "ul"
        start_attribute = ""
        if ordered:
            start_value = int(re.match(r"\d+", first.group("marker")).group(0))
            if start_value != 1:
                start_attribute = f' start="{start_value}"'

        items: list[str] = []
        while index < stop:
            current = _LIST_RE.match(lines[index])
            if current is None or _indent_width(current.group("space")) != base_indent:
                break
            current_ordered = current.group("marker")[0].isdigit()
            if current_ordered != ordered:
                break

            body_parts = [current.group("body").strip()]
            nested_parts: list[str] = []
            index += 1
            while index < stop:
                candidate = lines[index]
                candidate_list = _LIST_RE.match(candidate)
                if candidate_list:
                    candidate_indent = _indent_width(candidate_list.group("space"))
                    if candidate_indent <= base_indent:
                        break
                    nested, index = self._render_list(
                        lines,
                        index,
                        stop,
                        candidate_indent,
                    )
                    nested_parts.append(nested)
                    continue

                if not candidate.strip():
                    next_index = index + 1
                    while next_index < stop and not lines[next_index].strip():
                        next_index += 1
                    if next_index < stop:
                        next_list = _LIST_RE.match(lines[next_index])
                        next_indent = _indent_width(
                            lines[next_index][: len(lines[next_index]) - len(lines[next_index].lstrip())]
                        )
                        if (next_list and _indent_width(next_list.group("space")) > base_indent) or (
                            not next_list and next_indent > base_indent
                        ):
                            index = next_index
                            continue
                    break

                prefix = candidate[: len(candidate) - len(candidate.lstrip())]
                if _indent_width(prefix) > base_indent:
                    body_parts.append(candidate.strip())
                    index += 1
                    continue
                break

            checkbox = ""
            task = re.match(r"^\[([ xX])\][ \t]+(.*)$", body_parts[0])
            if task:
                checked = " checked" if task.group(1).lower() == "x" else ""
                checkbox = f'<input type="checkbox" disabled{checked} aria-hidden="true"> '
                body_parts[0] = task.group(2)
            item_body = checkbox + self.render_inline(" ".join(body_parts))
            if nested_parts:
                item_body += "\n" + "\n".join(nested_parts)
            items.append(f"<li>{item_body}</li>")

        return f"<{tag}{start_attribute}>\n" + "\n".join(items) + f"\n</{tag}>", index

    def _is_table_start(self, lines: list[str], index: int, stop: int) -> bool:
        if index + 1 >= stop or "|" not in lines[index]:
            return False
        separator = self._split_table_row(lines[index + 1])
        return bool(separator) and all(_TABLE_SEPARATOR_RE.match(cell.strip()) for cell in separator)

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|") and not stripped.endswith(r"\|"):
            stripped = stripped[:-1]
        cells: list[str] = []
        current: list[str] = []
        escaped = False
        code_ticks = 0
        for character in stripped:
            if character == "`" and not escaped:
                code_ticks ^= 1
            if character == "|" and not escaped and code_ticks == 0:
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(character)
            escaped = character == "\\" and not escaped
            if character != "\\":
                escaped = False
        cells.append("".join(current).strip())
        return cells

    def _render_table(self, lines: list[str], index: int, stop: int) -> tuple[str, int]:
        headers = self._split_table_row(lines[index])
        separators = self._split_table_row(lines[index + 1])
        alignments: list[str] = []
        for separator in separators:
            stripped = separator.strip()
            if stripped.startswith(":") and stripped.endswith(":"):
                alignments.append("center")
            elif stripped.endswith(":"):
                alignments.append("right")
            else:
                alignments.append("left")
        index += 2
        rows: list[list[str]] = []
        while index < stop and lines[index].strip() and "|" in lines[index]:
            row = self._split_table_row(lines[index])
            if len(row) != len(headers):
                break
            rows.append(row)
            index += 1

        head_cells = "".join(
            f'<th class="align-{alignments[column]}">{self.render_inline(value)}</th>'
            for column, value in enumerate(headers)
        )
        body_rows = []
        for row in rows:
            cells = "".join(
                f'<td class="align-{alignments[column]}">{self.render_inline(value)}</td>'
                for column, value in enumerate(row)
            )
            body_rows.append(f"<tr>{cells}</tr>")
        body = "\n".join(body_rows)
        return (
            '<div class="table-wrap"><table>\n'
            f"<thead><tr>{head_cells}</tr></thead>\n"
            f"<tbody>\n{body}\n</tbody>\n"
            "</table></div>",
            index,
        )

    def render_inline(self, value: str) -> str:
        output: list[str] = []
        index = 0
        literal_start = 0

        def flush(end: int) -> None:
            nonlocal literal_start
            if end > literal_start:
                output.append(html.escape(value[literal_start:end]))

        while index < len(value):
            if value[index] == "\\" and index + 1 < len(value):
                flush(index)
                output.append(html.escape(value[index + 1]))
                index += 2
                literal_start = index
                continue

            if value[index] == "`":
                count = len(value[index:]) - len(value[index:].lstrip("`"))
                delimiter = "`" * count
                closing = value.find(delimiter, index + count)
                if closing != -1:
                    flush(index)
                    code = value[index + count : closing]
                    if code.startswith(" ") and code.endswith(" ") and code.strip():
                        code = code[1:-1]
                    output.append(f"<code>{html.escape(code)}</code>")
                    index = closing + count
                    literal_start = index
                    continue

            if value.startswith("![", index) or value[index] == "[":
                image = value.startswith("![", index)
                bracket = index + 1 if image else index
                closing_bracket = self._find_balanced(value, bracket, "[", "]")
                if closing_bracket != -1 and closing_bracket + 1 < len(value) and value[closing_bracket + 1] == "(":
                    closing_parenthesis = self._find_balanced(
                        value,
                        closing_bracket + 1,
                        "(",
                        ")",
                    )
                    if closing_parenthesis != -1:
                        flush(index)
                        label = value[bracket + 1 : closing_bracket]
                        raw_destination = value[closing_bracket + 2 : closing_parenthesis]
                        destination, title = _split_destination(raw_destination)
                        rewritten = self._rewrite_destination(destination, image=image)
                        safe_destination = html.escape(rewritten, quote=True)
                        title_attribute = (
                            f' title="{html.escape(title, quote=True)}"' if title is not None else ""
                        )
                        if image:
                            alt = _html_to_text(self.render_inline(label))
                            output.append(
                                f'<img src="{safe_destination}" alt="{html.escape(alt, quote=True)}"'
                                f'{title_attribute} loading="lazy">'
                            )
                        else:
                            external = urlsplit(rewritten).scheme in {"http", "https"}
                            relationship = ' rel="noopener noreferrer"' if external else ""
                            output.append(
                                f'<a href="{safe_destination}"{title_attribute}{relationship}>'
                                f"{self.render_inline(label)}</a>"
                            )
                        index = closing_parenthesis + 1
                        literal_start = index
                        continue

            math_delimiter = "$$" if value.startswith("$$", index) else "$" if value[index] == "$" else ""
            if math_delimiter:
                closing = self._find_unescaped(value, math_delimiter, index + len(math_delimiter))
                if closing != -1:
                    flush(index)
                    expression = value[index + len(math_delimiter) : closing]
                    output.append(
                        '<span class="math-inline" data-math="inline">\\('
                        f"{html.escape(expression)}"
                        "\\)</span>"
                    )
                    index = closing + len(math_delimiter)
                    literal_start = index
                    continue

            matched_emphasis = False
            for marker, tag in (("**", "strong"), ("__", "strong"), ("*", "em"), ("_", "em")):
                if not value.startswith(marker, index):
                    continue
                closing = self._find_unescaped(value, marker, index + len(marker))
                if closing <= index + len(marker):
                    continue
                if marker == "_" and index > 0 and value[index - 1].isalnum():
                    continue
                flush(index)
                inner = value[index + len(marker) : closing]
                output.append(f"<{tag}>{self.render_inline(inner)}</{tag}>")
                index = closing + len(marker)
                literal_start = index
                matched_emphasis = True
                break
            if matched_emphasis:
                continue

            index += 1

        flush(len(value))
        return "".join(output)

    @staticmethod
    def _find_unescaped(value: str, needle: str, start: int) -> int:
        index = start
        while True:
            index = value.find(needle, index)
            if index == -1:
                return -1
            slashes = 0
            cursor = index - 1
            while cursor >= 0 and value[cursor] == "\\":
                slashes += 1
                cursor -= 1
            if slashes % 2 == 0:
                return index
            index += len(needle)

    @staticmethod
    def _find_balanced(value: str, start: int, opening: str, closing: str) -> int:
        if start >= len(value) or value[start] != opening:
            return -1
        depth = 0
        escaped = False
        for index in range(start, len(value)):
            character = value[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    return index
        return -1

    def _rewrite_destination(self, destination: str, *, image: bool) -> str:
        destination = destination.strip()
        if not destination:
            return "#"
        if destination.startswith("#"):
            return destination

        parsed = urlsplit(destination)
        scheme = parsed.scheme.lower()
        if scheme:
            if scheme not in {"http", "https", "mailto"}:
                return "#"
            return destination
        if parsed.netloc:
            return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))

        relative_path = parsed.path
        target = (self.source.parent / relative_path).resolve()
        guide_url = self.guide_urls.get(target)
        suffix = (f"?{parsed.query}" if parsed.query else "") + (
            f"#{parsed.fragment}" if parsed.fragment else ""
        )
        if guide_url:
            return guide_url + suffix

        try:
            repository_relative = target.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return "#"
        if target.exists():
            encoded = "/".join(quote(part) for part in repository_relative.parts)
            if image:
                return f"https://raw.githubusercontent.com/quangng2000/riftco-transformer/main/{encoded}{suffix}"
            return f"{REPOSITORY_BLOB_URL}{encoded}{suffix}"
        return destination


def _document_paths() -> list[Path]:
    paths = [PROJECT_ROOT / "README.md"]
    paths.extend(sorted(DOCS_SOURCE.glob("*.md"), key=lambda path: path.name.casefold()))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing documentation inputs: {', '.join(missing)}")
    return paths


def _exact_keys(
    value: Any,
    *,
    required: set[str],
    location: str,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{location} must be an object")
    optional = optional or set()
    keys = set(value)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required - optional)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise RuntimeError(f"{location} has invalid fields ({'; '.join(details)})")
    return value


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{location} must be a non-empty string")
    if value != value.strip():
        raise RuntimeError(f"{location} must not have leading or trailing whitespace")
    return value


def _required_order(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{location} must be a non-negative integer")
    return value


def _required_string_list(value: Any, location: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise RuntimeError(f"{location} must be {qualifier} of strings")
    values = tuple(_required_string(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(set(values)) != len(values):
        raise RuntimeError(f"{location} contains duplicates")
    return values


def _load_manifest() -> tuple[str, tuple[Section, ...], list[dict[str, Any]]]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"missing documentation manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    try:
        parsed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid documentation manifest JSON: {error}") from error

    manifest = _exact_keys(
        parsed,
        required={"version", "sections", "guides"},
        location="documentation manifest",
    )
    version = _required_string(manifest["version"], "manifest.version")

    if not isinstance(manifest["sections"], list) or not manifest["sections"]:
        raise RuntimeError("manifest.sections must be a non-empty array")
    sections: list[Section] = []
    for index, raw_section in enumerate(manifest["sections"]):
        section_value = _exact_keys(
            raw_section,
            required={"id", "title", "description", "order"},
            location=f"manifest.sections[{index}]",
        )
        identifier = _required_string(section_value["id"], f"manifest.sections[{index}].id")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", identifier):
            raise RuntimeError(
                f"manifest.sections[{index}].id must use lowercase letters, numbers, and interior hyphens"
            )
        sections.append(
            Section(
                identifier=identifier,
                title=_required_string(section_value["title"], f"manifest.sections[{index}].title"),
                description=_required_string(
                    section_value["description"],
                    f"manifest.sections[{index}].description",
                ),
                order=_required_order(section_value["order"], f"manifest.sections[{index}].order"),
            )
        )
    if len({section.identifier for section in sections}) != len(sections):
        raise RuntimeError("manifest.sections contains duplicate ids")
    if len({section.order for section in sections}) != len(sections):
        raise RuntimeError("manifest.sections contains duplicate order values")
    sections.sort(key=lambda section: (section.order, section.title.casefold(), section.identifier))

    if not isinstance(manifest["guides"], list) or not manifest["guides"]:
        raise RuntimeError("manifest.guides must be a non-empty array")
    guide_values: list[dict[str, Any]] = []
    required_guide_keys = {
        "source",
        "section",
        "type",
        "order",
        "summary",
        "stability",
        "backends",
        "audience",
    }
    expected_sources = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in _document_paths()
    }
    section_ids = {section.identifier for section in sections}
    seen_sources: set[str] = set()
    orders_by_section: dict[str, set[int]] = {identifier: set() for identifier in section_ids}

    for index, raw_guide in enumerate(manifest["guides"]):
        guide_value = _exact_keys(
            raw_guide,
            required=required_guide_keys,
            optional={"related"},
            location=f"manifest.guides[{index}]",
        )
        source_name = _required_string(guide_value["source"], f"manifest.guides[{index}].source")
        source_path = Path(source_name)
        if source_path.is_absolute() or ".." in source_path.parts or source_path.as_posix() != source_name:
            raise RuntimeError(f"manifest.guides[{index}].source must be a normalized repository path")
        if source_name not in expected_sources:
            raise RuntimeError(f"manifest.guides[{index}].source is not a documentation input: {source_name}")
        if source_name in seen_sources:
            raise RuntimeError(f"manifest.guides contains duplicate source: {source_name}")
        seen_sources.add(source_name)

        section_id = _required_string(guide_value["section"], f"manifest.guides[{index}].section")
        if section_id not in section_ids:
            raise RuntimeError(f"manifest.guides[{index}].section is unknown: {section_id}")
        guide_order = _required_order(guide_value["order"], f"manifest.guides[{index}].order")
        if guide_order in orders_by_section[section_id]:
            raise RuntimeError(
                f"manifest.guides has duplicate order {guide_order} in section {section_id}"
            )
        orders_by_section[section_id].add(guide_order)

        normalized = {
            "source": source_name,
            "section": section_id,
            "type": _required_string(guide_value["type"], f"manifest.guides[{index}].type"),
            "order": guide_order,
            "summary": _required_string(guide_value["summary"], f"manifest.guides[{index}].summary"),
            "stability": _required_string(
                guide_value["stability"],
                f"manifest.guides[{index}].stability",
            ),
            "backends": _required_string_list(
                guide_value["backends"],
                f"manifest.guides[{index}].backends",
            ),
            "audience": _required_string_list(
                guide_value["audience"],
                f"manifest.guides[{index}].audience",
            ),
            "related": _required_string_list(
                guide_value.get("related", []),
                f"manifest.guides[{index}].related",
                allow_empty=True,
            ),
        }
        guide_values.append(normalized)

    missing_sources = sorted(expected_sources - seen_sources)
    extra_sources = sorted(seen_sources - expected_sources)
    if missing_sources or extra_sources:
        details: list[str] = []
        if missing_sources:
            details.append("missing " + ", ".join(missing_sources))
        if extra_sources:
            details.append("unexpected " + ", ".join(extra_sources))
        raise RuntimeError("manifest must cover every documentation source exactly once (" + "; ".join(details) + ")")

    empty_sections = sorted(identifier for identifier, orders in orders_by_section.items() if not orders)
    if empty_sections:
        raise RuntimeError("manifest sections have no guides: " + ", ".join(empty_sections))

    for guide_value in guide_values:
        related_sources = guide_value["related"]
        invalid_related = sorted(set(related_sources) - expected_sources)
        if invalid_related:
            raise RuntimeError(
                f"manifest guide {guide_value['source']} has unknown related sources: "
                + ", ".join(invalid_related)
            )
        if guide_value["source"] in related_sources:
            raise RuntimeError(f"manifest guide {guide_value['source']} cannot relate to itself")

    return version, tuple(sections), guide_values


def _without_leading_document_title(content: str, headings: tuple[Heading, ...]) -> str:
    if not headings or headings[0].level != 1:
        return content
    return re.sub(r"\A\s*<h1\b.*?</h1>\s*", "", content, count=1, flags=re.DOTALL)


def _load_documentation() -> Documentation:
    version, sections, manifest_guides = _load_manifest()
    sections_by_id = {section.identifier: section for section in sections}

    guide_urls: dict[Path, str] = {}
    for metadata in manifest_guides:
        source = (PROJECT_ROOT / metadata["source"]).resolve()
        slug = _guide_slug(source)
        guide_urls[source] = f"../../docs/{metadata['section']}/{slug}.html"
    if len(set(guide_urls.values())) != len(guide_urls):
        raise RuntimeError("documentation filenames produce duplicate canonical guide paths")
    legacy_slugs = [_guide_slug(PROJECT_ROOT / metadata["source"]) for metadata in manifest_guides]
    if len(set(legacy_slugs)) != len(legacy_slugs):
        raise RuntimeError("documentation filenames produce duplicate legacy guide paths")

    guides: list[Guide] = []
    for metadata in manifest_guides:
        path = PROJECT_ROOT / metadata["source"]
        renderer = MarkdownRenderer(path, guide_urls)
        rendered_content = renderer.render(path.read_text(encoding="utf-8"))
        headings = tuple(renderer.headings)
        if not headings or headings[0].level != 1:
            raise RuntimeError(f"documentation source must start with a level-one heading: {metadata['source']}")
        title = headings[0].title
        content = _without_leading_document_title(rendered_content, headings)
        guides.append(
            Guide(
                source=path,
                source_name=metadata["source"],
                slug=_guide_slug(path),
                title=title,
                summary=metadata["summary"],
                section=sections_by_id[metadata["section"]],
                guide_type=metadata["type"],
                order=metadata["order"],
                stability=metadata["stability"],
                backends=metadata["backends"],
                audience=metadata["audience"],
                related_sources=metadata["related"],
                content=content,
                headings=headings,
                search_text=_normalise_space(f"{title} {metadata['summary']} {_html_to_text(content)}"),
            )
        )
    guides.sort(
        key=lambda guide: (
            guide.section.order,
            guide.order,
            guide.title.casefold(),
            guide.source_name,
        )
    )
    return Documentation(version=version, sections=sections, guides=tuple(guides))


def _site_href(site_root: str, site_path: str) -> str:
    return f"{site_root.rstrip('/')}/{site_path.lstrip('/')}"


def _guide_navigation(
    documentation: Documentation,
    *,
    current_section: Section,
    current_guide: Guide | None,
    site_root: str,
) -> str:
    groups: list[str] = []
    for section in documentation.sections:
        is_current_section = section == current_section
        details_classes = "guide-nav-group current" if is_current_section else "guide-nav-group"
        open_attribute = " open" if is_current_section else ""
        overview_classes = "guide-nav-section-link"
        overview_current = ""
        if is_current_section and current_guide is None:
            overview_classes += " active"
            overview_current = ' aria-current="page"'
        links = [
            f'<a class="{overview_classes}" href="{html.escape(_site_href(site_root, section.canonical_path), quote=True)}"{overview_current}>'
            "Section overview</a>"
        ]
        for guide in documentation.guides_in(section):
            classes = "active" if current_guide == guide else ""
            class_attribute = f' class="{classes}"' if classes else ""
            current_attribute = ' aria-current="page"' if current_guide == guide else ""
            links.append(
                f'<a href="{html.escape(_site_href(site_root, guide.canonical_path), quote=True)}"'
                f"{class_attribute}{current_attribute}>"
                f'<span>{html.escape(guide.title)}</span>'
                f'<small>{html.escape(guide.guide_type)}</small></a>'
            )
        groups.append(
            f'<details class="{details_classes}"{open_attribute}>'
            "<summary>"
            f'<span>{html.escape(section.title)}</span>'
            '<span class="guide-nav-chevron" aria-hidden="true">⌄</span>'
            "</summary>"
            '<div class="guide-nav-links">'
            + "\n".join(links)
            + "</div></details>"
        )
    return '<nav class="guide-nav" aria-label="Documentation sections">\n' + "\n".join(groups) + "\n</nav>"


def _table_of_contents(guide: Guide) -> str:
    headings = [heading for heading in guide.headings if 2 <= heading.level <= 3]
    if not headings:
        return '<p class="toc-empty">This guide has no subsections.</p>'
    links = [
        f'<a href="#{heading.identifier}" data-level="{heading.level}">'
        f"{html.escape(heading.title)}</a>"
        for heading in headings
    ]
    return '<nav aria-label="On this page">\n' + "\n".join(links) + "\n</nav>"


def _metadata_badges(guide: Guide) -> str:
    backend_badges = "".join(
        f'<span class="metadata-badge backend">{html.escape(backend)}</span>'
        for backend in guide.backends
    )
    audience = ", ".join(guide.audience)
    stability_class = _slugify(guide.stability, "status")
    return (
        '<dl class="guide-metadata">'
        '<div class="metadata-group"><dt class="metadata-label">Type</dt>'
        f'<dd class="metadata-value"><span class="metadata-badge type">{html.escape(guide.guide_type)}</span></dd></div>'
        '<div class="metadata-group"><dt class="metadata-label">Status</dt>'
        f'<dd class="metadata-value"><span class="metadata-badge stability {html.escape(stability_class, quote=True)}">'
        f'{html.escape(guide.stability)}</span></dd></div>'
        '<div class="metadata-group"><dt class="metadata-label">Backends</dt>'
        f'<dd class="metadata-value metadata-badges">{backend_badges}</dd></div>'
        '<div class="metadata-group"><dt class="metadata-label">For</dt>'
        f'<dd class="metadata-value">{html.escape(audience)}</dd></div>'
        "</dl>"
    )


def _guide_breadcrumbs(guide: Guide) -> str:
    return (
        '<nav class="docs-breadcrumbs" aria-label="Breadcrumb">'
        '<ol><li><a href="../../">Home</a></li>'
        '<li><a href="../">Docs</a></li>'
        f'<li><a href="index.html">{html.escape(guide.section.title)}</a></li>'
        f'<li aria-current="page">{html.escape(guide.title)}</li></ol></nav>'
    )


def _section_breadcrumbs(section: Section) -> str:
    return (
        '<nav class="docs-breadcrumbs" aria-label="Breadcrumb">'
        '<ol><li><a href="../../">Home</a></li>'
        '<li><a href="../">Docs</a></li>'
        f'<li aria-current="page">{html.escape(section.title)}</li></ol></nav>'
    )


def _related_guides(documentation: Documentation, current: Guide) -> tuple[Guide, ...]:
    by_source = {guide.source_name: guide for guide in documentation.guides}
    if current.related_sources:
        return tuple(by_source[source] for source in current.related_sources)

    def relevance(candidate: Guide) -> tuple[int, int, str, str]:
        score = 0
        if candidate.section == current.section:
            score += 12
        if candidate.guide_type.casefold() == current.guide_type.casefold():
            score += 4
        score += len(set(candidate.backends).intersection(current.backends))
        score += len(set(candidate.audience).intersection(current.audience))
        return (-score, abs(candidate.order - current.order), candidate.title.casefold(), candidate.source_name)

    candidates = sorted((guide for guide in documentation.guides if guide != current), key=relevance)
    return tuple(candidates[:3])


def _related_guide_cards(documentation: Documentation, current: Guide) -> str:
    cards = []
    for guide in _related_guides(documentation, current):
        cards.append(
            f'<a class="related-guide-card" href="../../{html.escape(guide.canonical_path, quote=True)}">'
            f'<span>{html.escape(guide.section.title)} · {html.escape(guide.guide_type)}</span>'
            f'<strong>{html.escape(guide.title)}</strong>'
            f'<small>{html.escape(guide.summary)}</small></a>'
        )
    return (
        '<section class="guide-related" aria-labelledby="related-guides-title">'
        '<div class="guide-related-heading"><span>Continue exploring</span>'
        '<h2 id="related-guides-title">Related documentation</h2></div>'
        '<div class="related-guide-grid">' + "\n".join(cards) + "</div></section>"
    )


def _adjacent_navigation(
    documentation: Documentation,
    position: int,
) -> tuple[str, str]:
    guides = documentation.guides
    previous = guides[position - 1] if position else None
    following = guides[position + 1] if position + 1 < len(guides) else None
    previous_link = (
        f'<a class="previous-guide" href="../../{html.escape(previous.canonical_path, quote=True)}">'
        '<span>Previous</span>'
        f'<strong>{html.escape(previous.title)}</strong>'
        f'<small>{html.escape(previous.section.title)}</small></a>'
        if previous
        else '<span class="previous-guide is-empty" aria-hidden="true"></span>'
    )
    following_link = (
        f'<a class="next-guide" href="../../{html.escape(following.canonical_path, quote=True)}">'
        '<span>Next</span>'
        f'<strong>{html.escape(following.title)}</strong>'
        f'<small>{html.escape(following.section.title)}</small></a>'
        if following
        else '<span class="next-guide is-empty" aria-hidden="true"></span>'
    )
    return previous_link, following_link


def _docs_section_cards(documentation: Documentation) -> str:
    cards: list[str] = []
    for index, section in enumerate(documentation.sections, start=1):
        guides = documentation.guides_in(section)
        types = sorted({guide.guide_type for guide in guides}, key=str.casefold)
        guide_links = "".join(
            f'<li><a href="{html.escape(section.identifier, quote=True)}/{html.escape(guide.slug, quote=True)}.html">'
            f'{html.escape(guide.title)} <span aria-hidden="true">→</span></a></li>'
            for guide in guides[:3]
        )
        remaining = len(guides) - min(3, len(guides))
        remaining_line = (
            f'<li class="docs-section-more">+ {remaining} more guide{"s" if remaining != 1 else ""}</li>'
            if remaining
            else ""
        )
        cards.append(
            '<article class="docs-section-card">'
            '<div class="docs-section-card-top">'
            f'<span class="docs-section-number">{index:02d}</span>'
            f'<span class="metadata-badge">{len(guides)} guide{"s" if len(guides) != 1 else ""}</span>'
            "</div>"
            f'<h2><a href="{html.escape(section.identifier, quote=True)}/">{html.escape(section.title)}</a></h2>'
            f'<p>{html.escape(section.description)}</p>'
            f'<div class="docs-section-meta">{html.escape(" · ".join(types))}</div>'
            f'<ul>{guide_links}{remaining_line}</ul>'
            f'<a class="docs-section-open" href="{html.escape(section.identifier, quote=True)}/">Browse section '
            '<span aria-hidden="true">→</span></a>'
            "</article>"
        )
    return "\n".join(cards)


def _documentation_type_summary(documentation: Documentation) -> str:
    counts: dict[str, int] = {}
    for guide in documentation.guides:
        counts[guide.guide_type] = counts.get(guide.guide_type, 0) + 1
    return "".join(
        '<div class="docs-type-stat">'
        f'<strong>{count}</strong><span>{html.escape(guide_type)}</span></div>'
        for guide_type, count in sorted(counts.items(), key=lambda item: item[0].casefold())
    )


def _footer_section_links(documentation: Documentation) -> str:
    return "".join(
        f'<a href="{html.escape(section.identifier, quote=True)}/">{html.escape(section.title)}</a>'
        for section in documentation.sections
    )


def _section_guide_cards(documentation: Documentation, section: Section) -> str:
    cards: list[str] = []
    for guide in documentation.guides_in(section):
        backend_badges = "".join(
            f'<span class="metadata-badge backend">{html.escape(backend)}</span>'
            for backend in guide.backends
        )
        cards.append(
            f'<article class="section-guide-card" data-guide-type="{html.escape(guide.guide_type, quote=True)}">'
            '<div class="section-guide-kicker">'
            f'<span>{html.escape(guide.guide_type)}</span>'
            f'<span class="metadata-badge stability {_slugify(guide.stability)}">{html.escape(guide.stability)}</span>'
            "</div>"
            f'<h2><a href="{html.escape(guide.slug, quote=True)}.html">{html.escape(guide.title)}</a></h2>'
            f'<p>{html.escape(guide.summary)}</p>'
            '<div class="section-guide-footer">'
            f'<span class="section-guide-audience">For {html.escape(", ".join(guide.audience))}</span>'
            f'<span class="metadata-badges">{backend_badges}</span>'
            "</div>"
            f'<a class="section-guide-open" href="{html.escape(guide.slug, quote=True)}.html" aria-label="Read {html.escape(guide.title, quote=True)}">'
            'Read guide <span aria-hidden="true">→</span></a>'
            "</article>"
        )
    return "\n".join(cards)


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    remaining = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if remaining:
        raise RuntimeError(f"unfilled guide template placeholders: {', '.join(remaining)}")
    return rendered


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_static_site(output: Path) -> None:
    for filename in ("index.html", "404.html"):
        page = SITE_SOURCE / filename
        if page.is_file():
            _copy_file(page, output / filename)

    assets = SITE_SOURCE / "assets"
    if assets.is_dir():
        for source in sorted((path for path in assets.rglob("*") if path.is_file()), key=lambda path: path.as_posix()):
            _copy_file(source, output / "assets" / source.relative_to(assets))

    open_graph_image = SITE_SOURCE / "og.png"
    if open_graph_image.is_file():
        _copy_file(open_graph_image, output / "og.png")


def _safe_reset_output(output: Path) -> None:
    resolved = output.resolve()
    forbidden = {
        Path("/").resolve(),
        PROJECT_ROOT.resolve(),
        PROJECT_ROOT.parent.resolve(),
        SITE_SOURCE.resolve(),
    }
    if resolved in forbidden or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing to replace unsafe output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _read_template(filename: str) -> str:
    path = TEMPLATES_DIRECTORY / filename
    if not path.is_file():
        raise RuntimeError(f"missing site template: {path.relative_to(PROJECT_ROOT)}")
    return path.read_text(encoding="utf-8")


def build_site(output: Path, base_url: str = DEFAULT_BASE_URL) -> Documentation:
    output = output.resolve()
    _safe_reset_output(output)
    guides_directory = output / "guides"
    guides_directory.mkdir(parents=True, exist_ok=True)
    docs_directory = output / "docs"
    docs_directory.mkdir(parents=True, exist_ok=True)
    _copy_static_site(output)

    guide_template = _read_template("guide.html")
    docs_home_template = _read_template("docs_home.html")
    section_template = _read_template("section.html")
    redirect_template = _read_template("redirect.html")
    documentation = _load_documentation()
    normalised_base_url = base_url.rstrip("/") + "/"

    docs_home = _render_template(
        docs_home_template,
        {
            "VERSION": html.escape(documentation.version),
            "VERSION_ATTRIBUTE": html.escape(documentation.version, quote=True),
            "GUIDE_COUNT": str(len(documentation.guides)),
            "SECTION_COUNT": str(len(documentation.sections)),
            "SECTION_CARDS": _docs_section_cards(documentation),
            "TYPE_SUMMARY": _documentation_type_summary(documentation),
            "FIRST_SECTION_TITLE": html.escape(documentation.sections[0].title),
            "FIRST_SECTION_URL_ATTRIBUTE": html.escape(
                documentation.sections[0].identifier,
                quote=True,
            ),
            "FOOTER_SECTION_LINKS": _footer_section_links(documentation),
            "CANONICAL_URL_ATTRIBUTE": html.escape(f"{normalised_base_url}docs/", quote=True),
            "OG_IMAGE_URL_ATTRIBUTE": html.escape(f"{normalised_base_url}og.png", quote=True),
        },
    )
    (docs_directory / "index.html").write_text(docs_home, encoding="utf-8", newline="\n")

    for section in documentation.sections:
        section_directory = docs_directory / section.identifier
        section_directory.mkdir(parents=True, exist_ok=True)
        section_page = _render_template(
            section_template,
            {
                "VERSION": html.escape(documentation.version),
                "VERSION_ATTRIBUTE": html.escape(documentation.version, quote=True),
                "TITLE": html.escape(section.title),
                "TITLE_ATTRIBUTE": html.escape(section.title, quote=True),
                "DESCRIPTION": html.escape(section.description),
                "DESCRIPTION_ATTRIBUTE": html.escape(section.description, quote=True),
                "GUIDE_COUNT": str(len(documentation.guides_in(section))),
                "GUIDE_NOUN": (
                    "guide" if len(documentation.guides_in(section)) == 1 else "guides"
                ),
                "BREADCRUMBS": _section_breadcrumbs(section),
                "GUIDE_NAVIGATION": _guide_navigation(
                    documentation,
                    current_section=section,
                    current_guide=None,
                    site_root="../..",
                ),
                "GUIDE_CARDS": _section_guide_cards(documentation, section),
                "CANONICAL_URL_ATTRIBUTE": html.escape(
                    f"{normalised_base_url}docs/{section.identifier}/",
                    quote=True,
                ),
                "OG_IMAGE_URL_ATTRIBUTE": html.escape(f"{normalised_base_url}og.png", quote=True),
            },
        )
        (section_directory / "index.html").write_text(
            section_page,
            encoding="utf-8",
            newline="\n",
        )

    for position, guide in enumerate(documentation.guides):
        previous_link, next_link = _adjacent_navigation(documentation, position)
        encoded_source = "/".join(quote(part) for part in Path(guide.source_name).parts)
        canonical_url = f"{normalised_base_url}{guide.canonical_path}"
        issue_title = quote(f"Docs: {guide.title}", safe="")
        issue_body = quote(
            f"Documentation page: {canonical_url}\nSource: {guide.source_name}\n\nDescribe the problem or suggested improvement:\n",
            safe="",
        )
        page = _render_template(
            guide_template,
            {
                "VERSION": html.escape(documentation.version),
                "VERSION_ATTRIBUTE": html.escape(documentation.version, quote=True),
                "TITLE": html.escape(guide.title),
                "TITLE_ATTRIBUTE": html.escape(guide.title, quote=True),
                "TITLE_ANCHOR_ATTRIBUTE": html.escape(guide.title_anchor, quote=True),
                "DESCRIPTION_ATTRIBUTE": html.escape(guide.summary, quote=True),
                "SUMMARY": html.escape(guide.summary),
                "SECTION_TITLE": html.escape(guide.section.title),
                "GUIDE_TYPE": html.escape(guide.guide_type),
                "CANONICAL_URL_ATTRIBUTE": html.escape(canonical_url, quote=True),
                "OG_IMAGE_URL_ATTRIBUTE": html.escape(f"{normalised_base_url}og.png", quote=True),
                "BREADCRUMBS": _guide_breadcrumbs(guide),
                "GUIDE_METADATA": _metadata_badges(guide),
                "SOURCE_PATH": html.escape(guide.source_name),
                "SOURCE_URL_ATTRIBUTE": html.escape(
                    f"{REPOSITORY_BLOB_URL}{encoded_source}",
                    quote=True,
                ),
                "EDIT_URL_ATTRIBUTE": html.escape(
                    f"{REPOSITORY_EDIT_URL}{encoded_source}",
                    quote=True,
                ),
                "ISSUE_URL_ATTRIBUTE": html.escape(
                    f"{REPOSITORY_ISSUES_URL}?title={issue_title}&body={issue_body}",
                    quote=True,
                ),
                "GUIDE_NAVIGATION": _guide_navigation(
                    documentation,
                    current_section=guide.section,
                    current_guide=guide,
                    site_root="../..",
                ),
                "TABLE_OF_CONTENTS": _table_of_contents(guide),
                "CONTENT": guide.content,
                "RELATED_GUIDES": _related_guide_cards(documentation, guide),
                "PREVIOUS_GUIDE": previous_link,
                "NEXT_GUIDE": next_link,
            },
        )
        canonical_output = output / guide.canonical_path
        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        canonical_output.write_text(page, encoding="utf-8", newline="\n")

        redirect_target = f"../{guide.canonical_path}"
        redirect_page = _render_template(
            redirect_template,
            {
                "TITLE": html.escape(guide.title),
                "TITLE_ATTRIBUTE": html.escape(guide.title, quote=True),
                "CANONICAL_URL_ATTRIBUTE": html.escape(canonical_url, quote=True),
                "REDIRECT_TARGET_ATTRIBUTE": html.escape(redirect_target, quote=True),
                "REDIRECT_TARGET_JSON": json.dumps(redirect_target, ensure_ascii=False),
            },
        )
        (output / guide.legacy_path).write_text(
            redirect_page,
            encoding="utf-8",
            newline="\n",
        )

    search_index = [
        {
            "audience": list(guide.audience),
            "backends": list(guide.backends),
            "description": guide.summary,
            "headings": [
                {
                    "id": heading.identifier,
                    "level": heading.level,
                    "title": heading.title,
                }
                for heading in guide.headings
            ],
            "section": guide.section.identifier,
            "section_title": guide.section.title,
            "source": guide.source_name,
            "stability": guide.stability,
            "text": guide.search_text,
            "title": guide.title,
            "type": guide.guide_type,
            "url": guide.canonical_path,
            "version": documentation.version,
        }
        for guide in documentation.guides
    ]
    (output / "search-index.json").write_text(
        json.dumps(search_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    locations = [
        normalised_base_url,
        f"{normalised_base_url}docs/",
        *(f"{normalised_base_url}docs/{section.identifier}/" for section in documentation.sections),
        *(f"{normalised_base_url}{guide.canonical_path}" for guide in documentation.guides),
    ]
    sitemap_urls = "\n".join(
        f"  <url><loc>{html.escape(location)}</loc></url>" for location in locations
    )
    (output / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{sitemap_urls}\n"
        "</urlset>\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "robots.txt").write_text(
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {normalised_base_url}sitemap.xml\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return documentation


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=lambda path: path.as_posix())
    }


def _check_internal_links(output: Path, base_url: str) -> None:
    html_files = sorted(output.rglob("*.html"), key=lambda path: path.as_posix())
    cached_pages = {
        path.resolve(): path.read_text(encoding="utf-8") for path in html_files
    }
    public_path = urlsplit(base_url).path.rstrip("/") + "/"
    failures: list[str] = []
    for page in html_files:
        contents = cached_pages[page.resolve()]
        for encoded_href in re.findall(r'href="([^"]*)"', contents):
            href = html.unescape(encoded_href)
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or href.startswith("mailto:") or href == "#":
                continue
            decoded_path = unquote(parsed.path)
            if decoded_path.startswith(public_path):
                site_relative = decoded_path[len(public_path) :]
                target = (output / site_relative).resolve() if site_relative else output.resolve()
            else:
                target = (page.parent / decoded_path).resolve() if decoded_path else page.resolve()
            if decoded_path.endswith("/") or target.is_dir():
                target /= "index.html"
            if target.suffix.lower() != ".html":
                continue
            if target not in cached_pages:
                failures.append(f"{page.relative_to(output)} -> {href} (missing page)")
                continue
            redirect_match = re.search(
                r'data-redirect-target="([^"]+)"',
                cached_pages[target],
            )
            anchor_target = target
            if redirect_match:
                redirected_href = html.unescape(redirect_match.group(1))
                redirected_path = unquote(urlsplit(redirected_href).path)
                anchor_target = (target.parent / redirected_path).resolve()
                if redirected_path.endswith("/") or anchor_target.is_dir():
                    anchor_target /= "index.html"
                if anchor_target not in cached_pages:
                    failures.append(
                        f"{target.relative_to(output)} redirects to {redirected_href} (missing page)"
                    )
                    continue
            if parsed.fragment:
                identifier = html.escape(unquote(parsed.fragment), quote=True)
                if f'id="{identifier}"' not in cached_pages[anchor_target]:
                    failures.append(f"{page.relative_to(output)} -> {href} (missing anchor)")
    if failures:
        raise RuntimeError("broken internal links:\n" + "\n".join(failures))


def check_site(output: Path, documentation: Documentation, base_url: str) -> None:
    expected = {
        ".nojekyll",
        "404.html",
        "assets/app.js",
        "assets/styles.css",
        "index.html",
        "og.png",
        "robots.txt",
        "search-index.json",
        "sitemap.xml",
        "docs/index.html",
        *(section.canonical_path for section in documentation.sections),
        *(guide.canonical_path for guide in documentation.guides),
        *(guide.legacy_path for guide in documentation.guides),
    }
    actual = set(_tree_digests(output))
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"generated site is missing: {', '.join(missing)}")

    for page in sorted(output.rglob("*.html"), key=lambda path: path.as_posix()):
        contents = page.read_text(encoding="utf-8")
        placeholders = sorted(set(_PLACEHOLDER_RE.findall(contents)))
        if placeholders:
            raise RuntimeError(
                f"{page.relative_to(output)} has unfilled placeholders: {', '.join(placeholders)}"
            )

    normalised_base_url = base_url.rstrip("/") + "/"
    for guide in documentation.guides:
        page = output / guide.canonical_path
        contents = page.read_text(encoding="utf-8")
        if 'data-site-root="../.."' not in contents:
            raise RuntimeError(
                f"{guide.canonical_path} does not declare data-site-root='../..'"
            )
        h1_count = len(re.findall(r"<h1(?:\s|>)", contents))
        if h1_count != 1:
            raise RuntimeError(
                f"{guide.canonical_path} must contain exactly one h1; found {h1_count}"
            )
        if f'<h1 id="{html.escape(guide.title_anchor, quote=True)}">' not in contents:
            raise RuntimeError(f"{guide.canonical_path} does not preserve its title anchor")
        required_metadata = (
            guide.guide_type,
            guide.stability,
            *guide.backends,
            *guide.audience,
        )
        missing_metadata = [value for value in required_metadata if html.escape(value) not in contents]
        if missing_metadata:
            raise RuntimeError(
                f"{guide.canonical_path} is missing guide metadata: {', '.join(missing_metadata)}"
            )
        if "Edit page" not in contents or "Report issue" not in contents:
            raise RuntimeError(f"{guide.canonical_path} is missing documentation feedback actions")

        redirect = output / guide.legacy_path
        redirect_contents = redirect.read_text(encoding="utf-8")
        expected_target = f"../{guide.canonical_path}"
        if f'data-redirect-target="{expected_target}"' not in redirect_contents:
            raise RuntimeError(f"{guide.legacy_path} does not redirect to {guide.canonical_path}")
        expected_canonical = f"{normalised_base_url}{guide.canonical_path}"
        if f'<link rel="canonical" href="{expected_canonical}">' not in redirect_contents:
            raise RuntimeError(f"{guide.legacy_path} has the wrong canonical URL")

    docs_home = (output / "docs" / "index.html").read_text(encoding="utf-8")
    if 'data-site-root=".."' not in docs_home:
        raise RuntimeError("docs/index.html does not declare data-site-root='..'")
    for section in documentation.sections:
        section_page = (output / section.canonical_path).read_text(encoding="utf-8")
        if 'data-site-root="../.."' not in section_page:
            raise RuntimeError(
                f"{section.canonical_path} does not declare data-site-root='../..'"
            )

    parsed_index = json.loads((output / "search-index.json").read_text(encoding="utf-8"))
    if not isinstance(parsed_index, list) or len(parsed_index) != len(documentation.guides):
        raise RuntimeError("search index entry count does not match guide count")
    required_search_keys = {
        "audience",
        "backends",
        "description",
        "headings",
        "section",
        "section_title",
        "source",
        "stability",
        "text",
        "title",
        "type",
        "url",
        "version",
    }
    for index, entry in enumerate(parsed_index):
        if not isinstance(entry, dict) or set(entry) != required_search_keys:
            raise RuntimeError(f"search index entry {index} does not match the public search schema")
    indexed_urls = {entry["url"] for entry in parsed_index}
    expected_urls = {guide.canonical_path for guide in documentation.guides}
    if indexed_urls != expected_urls:
        raise RuntimeError("search index URLs do not match generated guides")
    indexed_by_source = {entry["source"]: entry for entry in parsed_index}
    for guide in documentation.guides:
        entry = indexed_by_source.get(guide.source_name)
        if not entry:
            raise RuntimeError(f"search index is missing {guide.source_name}")
        if entry["section"] != guide.section.identifier or entry["type"] != guide.guide_type:
            raise RuntimeError(f"search index metadata does not match {guide.source_name}")
        guide_page = (output / guide.canonical_path).read_text(encoding="utf-8")
        for heading in entry["headings"]:
            identifier = html.escape(str(heading["id"]), quote=True)
            if f'id="{identifier}"' not in guide_page:
                raise RuntimeError(
                    f"search index anchor {heading['id']} is missing from {guide.canonical_path}"
                )

    sitemap_contents = (output / "sitemap.xml").read_text(encoding="utf-8")
    sitemap_locations = [html.unescape(location) for location in re.findall(r"<loc>(.*?)</loc>", sitemap_contents)]
    expected_locations = [
        normalised_base_url,
        f"{normalised_base_url}docs/",
        *(f"{normalised_base_url}docs/{section.identifier}/" for section in documentation.sections),
        *(f"{normalised_base_url}{guide.canonical_path}" for guide in documentation.guides),
    ]
    if sitemap_locations != expected_locations:
        raise RuntimeError("sitemap does not contain exactly the canonical documentation pages")
    if any("/guides/" in location for location in sitemap_locations):
        raise RuntimeError("sitemap must not include legacy guide redirects")

    _check_internal_links(output, base_url)

    with tempfile.TemporaryDirectory(prefix="riftco-pages-check-") as temporary:
        comparison = Path(temporary) / "site"
        build_site(comparison, base_url)
        if _tree_digests(output) != _tree_digests(comparison):
            raise RuntimeError("site generation is not byte-for-byte deterministic")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="public site URL used by canonical links, sitemap.xml, and robots.txt",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate completeness and rebuild once to prove deterministic output",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    documentation = build_site(arguments.output, arguments.base_url)
    if arguments.check:
        check_site(arguments.output.resolve(), documentation, arguments.base_url)
    print(
        f"built {len(documentation.guides)} guides across {len(documentation.sections)} sections "
        f"for v{documentation.version} in {arguments.output.resolve()}"
        + (" (checks passed)" if arguments.check else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
