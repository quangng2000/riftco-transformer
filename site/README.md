# Documentation site

The GitHub Pages site is intentionally a static, dependency-free build. The
landing page and browser assets live in this directory. `docs_manifest.json`
classifies the maintained `README.md` and every `docs/*.md` file by reader
intent, content type, stability, backend scope, and audience.

The build publishes a documentation home, eight section indexes, canonical
guide pages under `docs/<section>/`, and redirects for the original
`guides/*.html` URLs. Keep the manifest coverage exact whenever a guide is
added, moved between sections, or retired.

Build and validate it locally:

```bash
python3 .github/scripts/build_pages.py --output build/pages --check
python3 -m http.server 8000 --directory build/pages
```

Then open `http://127.0.0.1:8000/docs/`. The builder validates manifest
coverage, canonical and legacy routes, internal pages and anchors, metadata,
search coverage, the sitemap, and byte-for-byte determinism. Pushes to `main`
that touch documentation or site files run
`.github/workflows/pages.yml` and publish the resulting artifact to GitHub
Pages.

The generated guides use only repository code at build time. In the browser,
MathJax and Mermaid are loaded from pinned CDN versions when a page contains
math or a diagram; the escaped source remains readable if either optional
renderer is unavailable.
