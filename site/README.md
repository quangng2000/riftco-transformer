# Documentation site

The GitHub Pages site is intentionally a static, dependency-free build. The
landing page and browser assets live in this directory; the maintained
`README.md` and every `docs/*.md` file become searchable guide pages during the
build.

Build and validate it locally:

```bash
python3 .github/scripts/build_pages.py --output build/pages --check
python3 -m http.server 8000 --directory build/pages
```

Then open `http://127.0.0.1:8000/`. The builder validates generated files,
internal page and anchor links, search-index coverage, and byte-for-byte
determinism. Pushes to `main` that touch documentation or site files run
`.github/workflows/pages.yml` and publish the resulting artifact to GitHub
Pages.

The generated guides use only repository code at build time. In the browser,
MathJax and Mermaid are loaded from pinned CDN versions when a page contains
math or a diagram; the escaped source remains readable if either optional
renderer is unavailable.
