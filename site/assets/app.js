(() => {
  "use strict";

  const doc = document.documentElement;
  const body = document.body;
  const root = body.dataset.siteRoot || ".";

  const storedTheme = localStorage.getItem("riftco-theme");
  const preferredTheme = window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
  doc.dataset.theme = storedTheme || preferredTheme;

  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = doc.dataset.theme === "dark" ? "light" : "dark";
      doc.dataset.theme = next;
      localStorage.setItem("riftco-theme", next);
      button.setAttribute("aria-label", `Switch to ${next === "dark" ? "light" : "dark"} theme`);
    });
  });

  const header = document.querySelector("[data-header]");
  const progress = document.querySelector(".reading-progress span");
  const updateScrollState = () => {
    header?.classList.toggle("scrolled", window.scrollY > 8);
    if (!progress) return;
    const available = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = available > 0 ? Math.min(1, window.scrollY / available) : 0;
    progress.style.width = `${ratio * 100}%`;
  };
  updateScrollState();
  window.addEventListener("scroll", updateScrollState, { passive: true });

  const navToggle = document.querySelector("[data-nav-toggle]");
  const siteNav = document.querySelector("[data-site-nav]");
  navToggle?.addEventListener("click", () => {
    const open = siteNav?.classList.toggle("open") || false;
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  siteNav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      siteNav.classList.remove("open");
      navToggle?.setAttribute("aria-expanded", "false");
    });
  });

  document.querySelectorAll("[data-code-tabs]").forEach((group) => {
    const tabs = Array.from(group.querySelectorAll("[data-code-tab]"));
    const panels = Array.from(group.querySelectorAll("[data-code-panel]"));
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.codeTab;
        tabs.forEach((candidate) => {
          candidate.setAttribute("aria-selected", String(candidate === tab));
        });
        panels.forEach((panel) => {
          const active = panel.dataset.codePanel === target;
          panel.hidden = !active;
          panel.classList.toggle("active", active);
        });
      });
    });
  });

  document.querySelectorAll("pre").forEach((pre) => {
    if (pre.closest(".mermaid") || pre.querySelector(".copy-code")) return;
    const code = pre.querySelector("code");
    if (!code) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || "");
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = "Copy"; }, 1400);
      } catch {
        button.textContent = "Select";
        const range = document.createRange();
        range.selectNodeContents(code);
        window.getSelection()?.removeAllRanges();
        window.getSelection()?.addRange(range);
      }
    });
    pre.append(button);
  });

  const reveals = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("revealed");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12 });
    reveals.forEach((element) => observer.observe(element));
  } else {
    reveals.forEach((element) => element.classList.add("revealed"));
  }

  const tocLinks = Array.from(document.querySelectorAll(".guide-toc a[href^='#']"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    const headingMap = new Map();
    tocLinks.forEach((link) => {
      const heading = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (heading) headingMap.set(heading, link);
    });
    const tocObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        tocLinks.forEach((link) => link.classList.remove("active"));
        headingMap.get(entry.target)?.classList.add("active");
      });
    }, { rootMargin: "-15% 0px -72% 0px" });
    headingMap.forEach((_, heading) => tocObserver.observe(heading));
  }

  const guideMenu = document.querySelector("[data-guide-menu]");
  const guideSidebar = document.querySelector(".guide-sidebar");
  guideMenu?.addEventListener("click", () => {
    const open = guideSidebar?.classList.toggle("open") || false;
    guideMenu.setAttribute("aria-expanded", String(open));
  });

  const dialog = document.querySelector("[data-search-dialog]");
  const input = dialog?.querySelector("[data-search-input]");
  const results = dialog?.querySelector("[data-search-results]");
  let searchIndex = [];
  let activeResult = -1;

  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

  const scoreEntry = (entry, terms) => {
    const title = entry.title.toLowerCase();
    const description = (entry.description || "").toLowerCase();
    const headings = (entry.headings || []).map((heading) => heading.title).join(" ").toLowerCase();
    const text = (entry.text || "").toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (!title.includes(term) && !description.includes(term) && !headings.includes(term) && !text.includes(term)) return -1;
      if (title === term) score += 30;
      else if (title.includes(term)) score += 15;
      if (headings.includes(term)) score += 6;
      if (description.includes(term)) score += 4;
      if (text.includes(term)) score += 1;
    }
    return score;
  };

  const renderResults = () => {
    if (!input || !results) return;
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    activeResult = -1;
    if (!terms.length) {
      results.innerHTML = '<div class="search-empty"><strong>Search the complete framework</strong><span>Try “strides”, “paged Adam”, “Metal”, or “compiled attention”.</span></div>';
      return;
    }
    const matches = searchIndex
      .map((entry) => ({ entry, score: scoreEntry(entry, terms) }))
      .filter((item) => item.score >= 0)
      .sort((left, right) => right.score - left.score || left.entry.title.localeCompare(right.entry.title))
      .slice(0, 12);
    if (!matches.length) {
      results.innerHTML = `<div class="search-empty"><strong>No match for “${escapeHtml(input.value)}”</strong><span>Try a component, algorithm, or backend name.</span></div>`;
      return;
    }
    results.innerHTML = matches.map(({ entry }) => {
      const headingMatch = (entry.headings || []).find((heading) =>
        terms.some((term) => heading.title.toLowerCase().includes(term))
      );
      const suffix = headingMatch ? `#${headingMatch.id}` : "";
      const detail = headingMatch?.title || entry.description || entry.source;
      return `<a class="search-result" href="${root}/${entry.url}${suffix}"><strong>${escapeHtml(entry.title)}</strong><span>${escapeHtml(detail)}</span></a>`;
    }).join("");
  };

  const ensureSearchIndex = async () => {
    if (searchIndex.length) return;
    try {
      const response = await fetch(`${root}/search-index.json`);
      if (!response.ok) throw new Error(`search index returned ${response.status}`);
      searchIndex = await response.json();
    } catch {
      if (results) {
        results.innerHTML = '<div class="search-empty"><strong>Search is unavailable in this preview</strong><span>Browse the documentation navigation instead.</span></div>';
      }
    }
  };

  const openSearch = async () => {
    if (!(dialog instanceof HTMLDialogElement)) return;
    if (!dialog.open) dialog.showModal();
    await ensureSearchIndex();
    window.setTimeout(() => input?.focus(), 0);
  };

  const closeSearch = () => {
    if (dialog instanceof HTMLDialogElement && dialog.open) dialog.close();
  };

  document.querySelectorAll("[data-search-open]").forEach((button) => button.addEventListener("click", openSearch));
  dialog?.querySelector("[data-search-close]")?.addEventListener("click", closeSearch);
  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) closeSearch();
  });
  input?.addEventListener("input", renderResults);
  input?.addEventListener("keydown", (event) => {
    const links = Array.from(results?.querySelectorAll(".search-result") || []);
    if (event.key === "ArrowDown" && links.length) {
      event.preventDefault();
      activeResult = (activeResult + 1) % links.length;
    } else if (event.key === "ArrowUp" && links.length) {
      event.preventDefault();
      activeResult = (activeResult - 1 + links.length) % links.length;
    } else if (event.key === "Enter" && activeResult >= 0) {
      event.preventDefault();
      links[activeResult].click();
      return;
    } else {
      return;
    }
    links.forEach((link, index) => link.classList.toggle("active", index === activeResult));
    links[activeResult]?.scrollIntoView({ block: "nearest" });
  });

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const typing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target?.isContentEditable;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    } else if (event.key === "/" && !typing) {
      event.preventDefault();
      openSearch();
    } else if (event.key === "Escape") {
      closeSearch();
    }
  });
})();

(() => {
  "use strict";

  const math = document.querySelector("[data-math], .math-inline");
  if (math) {
    window.MathJax = {
      tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
      },
      options: {
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      },
    };
    const script = document.createElement("script");
    script.defer = true;
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@4.0.0/tex-chtml.js";
    script.dataset.docsRenderer = "mathjax";
    document.head.append(script);
  }

  const diagrams = document.querySelectorAll(".mermaid");
  if (diagrams.length) {
    import("https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.esm.min.mjs")
      .then(async ({ default: mermaid }) => {
        const dark = document.documentElement.dataset.theme !== "light";
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: dark ? "dark" : "neutral",
          fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          themeVariables: dark
            ? {
                background: "#151916",
                primaryColor: "#1a201c",
                primaryTextColor: "#f4f2e9",
                primaryBorderColor: "#c7ff4a",
                lineColor: "#858b82",
                secondaryColor: "#222a24",
                tertiaryColor: "#111412",
              }
            : undefined,
        });
        await mermaid.run({ nodes: diagrams, suppressErrors: true });
      })
      .catch(() => {
        // The escaped diagram source remains readable when the optional CDN
        // renderer is unavailable.
      });
  }
})();
