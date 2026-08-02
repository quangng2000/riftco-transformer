#!/usr/bin/env python3
"""Build the dependency-free Riftco Transformer documentation site.

The repository deliberately avoids a Python Markdown dependency.  This small
renderer implements the subset used by README.md and docs/*.md, escapes all
source text before placing it in HTML, and emits deterministic static files
that GitHub Pages can serve from a project subpath.
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
from typing import Iterable
from urllib.parse import quote, unquote, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_SOURCE = PROJECT_ROOT / "site"
DOCS_SOURCE = PROJECT_ROOT / "docs"
TEMPLATE_PATH = SITE_SOURCE / "templates" / "guide.html"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "pages"
DEFAULT_BASE_URL = "https://quangng2000.github.io/riftco-transformer/"
REPOSITORY_BLOB_URL = "https://github.com/quangng2000/riftco-transformer/blob/main/"

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
class Guide:
    source: Path
    source_name: str
    slug: str
    title: str
    description: str
    content: str
    headings: tuple[Heading, ...]
    search_text: str


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


def _description(content: str, title: str) -> str:
    for paragraph in re.findall(r"<p>(.*?)</p>", content, flags=re.DOTALL):
        text = _html_to_text(paragraph)
        if text and text != title:
            return text if len(text) <= 240 else text[:237].rstrip() + "..."
    return title


def _load_guides() -> list[Guide]:
    paths = _document_paths()
    guide_urls = {path.resolve(): f"{_guide_slug(path)}.html" for path in paths}
    if len(set(guide_urls.values())) != len(guide_urls):
        raise RuntimeError("documentation filenames produce duplicate guide slugs")

    guides: list[Guide] = []
    for path in paths:
        renderer = MarkdownRenderer(path, guide_urls)
        content = renderer.render(path.read_text(encoding="utf-8"))
        headings = tuple(renderer.headings)
        title = headings[0].title if headings else path.stem.replace("_", " ").title()
        guides.append(
            Guide(
                source=path,
                source_name=path.relative_to(PROJECT_ROOT).as_posix(),
                slug=_guide_slug(path),
                title=title,
                description=_description(content, title),
                content=content,
                headings=headings,
                search_text=_html_to_text(content),
            )
        )
    return guides


def _guide_navigation(guides: Iterable[Guide], current: Guide) -> str:
    links = []
    for guide in guides:
        current_attribute = (
            ' class="active" aria-current="page"' if guide.slug == current.slug else ""
        )
        links.append(
            f'<a href="{guide.slug}.html"{current_attribute}>'
            f"{html.escape(guide.title)}</a>"
        )
    return '<nav class="guide-nav" aria-label="Framework guides">\n' + "\n".join(links) + "\n</nav>"


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


def build_site(output: Path, base_url: str = DEFAULT_BASE_URL) -> list[Guide]:
    output = output.resolve()
    _safe_reset_output(output)
    guides_directory = output / "guides"
    guides_directory.mkdir(parents=True, exist_ok=True)
    _copy_static_site(output)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    guides = _load_guides()
    normalised_base_url = base_url.rstrip("/") + "/"

    for position, guide in enumerate(guides):
        previous_guide = guides[position - 1] if position else None
        next_guide = guides[position + 1] if position + 1 < len(guides) else None
        previous_link = (
            f'<a class="previous-guide" href="{previous_guide.slug}.html">'
            f'<span>Previous</span><strong>{html.escape(previous_guide.title)}</strong></a>'
            if previous_guide
            else '<span class="previous-guide is-empty" aria-hidden="true"></span>'
        )
        next_link = (
            f'<a class="next-guide" href="{next_guide.slug}.html">'
            f'<span>Next</span><strong>{html.escape(next_guide.title)}</strong></a>'
            if next_guide
            else '<span class="next-guide is-empty" aria-hidden="true"></span>'
        )
        encoded_source = "/".join(quote(part) for part in Path(guide.source_name).parts)
        page = _render_template(
            template,
            {
                "TITLE": html.escape(guide.title),
                "TITLE_ATTRIBUTE": html.escape(guide.title, quote=True),
                "DESCRIPTION_ATTRIBUTE": html.escape(guide.description, quote=True),
                "CANONICAL_URL_ATTRIBUTE": html.escape(
                    f"{normalised_base_url}guides/{guide.slug}.html",
                    quote=True,
                ),
                "OG_IMAGE_URL_ATTRIBUTE": html.escape(
                    f"{normalised_base_url}og.png",
                    quote=True,
                ),
                "SOURCE_PATH": html.escape(guide.source_name),
                "SOURCE_URL_ATTRIBUTE": html.escape(
                    f"{REPOSITORY_BLOB_URL}{encoded_source}",
                    quote=True,
                ),
                "GUIDE_NAVIGATION": _guide_navigation(guides, guide),
                "TABLE_OF_CONTENTS": _table_of_contents(guide),
                "CONTENT": guide.content,
                "PREVIOUS_GUIDE": previous_link,
                "NEXT_GUIDE": next_link,
            },
        )
        (guides_directory / f"{guide.slug}.html").write_text(page, encoding="utf-8", newline="\n")

    search_index = [
        {
            "description": guide.description,
            "headings": [
                {
                    "id": heading.identifier,
                    "level": heading.level,
                    "title": heading.title,
                }
                for heading in guide.headings
            ],
            "source": guide.source_name,
            "text": guide.search_text,
            "title": guide.title,
            "url": f"guides/{guide.slug}.html",
        }
        for guide in guides
    ]
    (output / "search-index.json").write_text(
        json.dumps(search_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    locations = [normalised_base_url] + [
        f"{normalised_base_url}guides/{guide.slug}.html" for guide in guides
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
    return guides


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
            if parsed.fragment:
                identifier = html.escape(unquote(parsed.fragment), quote=True)
                if f'id="{identifier}"' not in cached_pages[target]:
                    failures.append(f"{page.relative_to(output)} -> {href} (missing anchor)")
    if failures:
        raise RuntimeError("broken internal links:\n" + "\n".join(failures))


def check_site(output: Path, guides: list[Guide], base_url: str) -> None:
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
        *(f"guides/{guide.slug}.html" for guide in guides),
    }
    actual = set(_tree_digests(output))
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"generated site is missing: {', '.join(missing)}")

    for page in sorted((output / "guides").glob("*.html")):
        contents = page.read_text(encoding="utf-8")
        placeholders = sorted(set(_PLACEHOLDER_RE.findall(contents)))
        if placeholders:
            raise RuntimeError(f"{page.name} has unfilled placeholders: {', '.join(placeholders)}")
        if 'data-site-root=".."' not in contents:
            raise RuntimeError(f"{page.name} does not declare its project-relative site root")

    parsed_index = json.loads((output / "search-index.json").read_text(encoding="utf-8"))
    if len(parsed_index) != len(guides):
        raise RuntimeError("search index entry count does not match guide count")
    indexed_urls = {entry["url"] for entry in parsed_index}
    expected_urls = {f"guides/{guide.slug}.html" for guide in guides}
    if indexed_urls != expected_urls:
        raise RuntimeError("search index URLs do not match generated guides")

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
    guides = build_site(arguments.output, arguments.base_url)
    if arguments.check:
        check_site(arguments.output.resolve(), guides, arguments.base_url)
    print(
        f"built {len(guides)} guides in {arguments.output.resolve()}"
        + (" (checks passed)" if arguments.check else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
